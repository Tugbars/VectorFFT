# The K=1 Optimization Campaign — from 0.26× to MKL parity in one arc

*2026-07-22. Host: i9-14900KF (Raptor Lake, AVX2 only, 5.7 GHz), MKL 2025.3 sequential,
mingw-gcc 15.2, Windows 11. Companion doctrine doc (designs, disassembly, decisions):
[docs/roadmap/row_major_engine.md](../roadmap/row_major_engine.md) §11–§13.3. This document
is the performance narrative: every optimization applied, its mechanism, its measured gain,
and the negative results that shaped the path.*

---

## 1. The starting point

A single transform (`howmany = 1` — the most common FFT call in the world, and what every
MKL/FFTW user issues by default) had no native path in the engine. The whole codelet arsenal
vectorizes across the **batch** dimension K; at K=1 there is no batch, so execution fell to
the scalar tier (`vl=1`, one SIMD lane doing all the work) or to the half-vectorized BAILEY2.

Measured starting positions (hot-loop, pinned, best-of-5):

| N | scalar tier | old best route | MKL-split | MKL-IL | our best vs MKL-IL |
|--:|--:|--:|--:|--:|--:|
| 64 | 151–164 | LEAF 151 | 33.7 | 30.2 | 0.20× |
| 256 | 799 | BAILEY2 382 | 194 | 136 | 0.36× |
| 1024 | 3585 | BAILEY2 2442 | 740–766 | 750–874 | 0.30× |
| 4096 | 18 430 | BAILEY2 16 220 | 4907 | 3980 | 0.25× |

Roughly **a quarter of MKL's speed** — hence this campaign. The end state, one day later:
**parity with MKL-IL at 64, wins over MKL-split through 256, MKL-split parity at 1024**, and
0.6–0.8× at the largest sizes with the remaining levers identified and scoped.

## 2. Ground truth before optimization

Nothing below was guessed. Three evidence-gathering efforts preceded and steered the work:

### 2.1 MKL disassembly (gdb hardware watchpoints on the output buffer)

- **N=64 (interleaved)**: one fused in-register kernel, ~616 instructions, zero loops,
  rip-relative twiddle constants. The "mono tier."
- **N=64 (split/REAL_REAL)**: a separate kernel with **zero re/im shuffles** — MKL covers
  both layouts natively; there is no free win from "avoiding their shuffle tax."
- **N=256/512/1024**: exactly **TWO passes** — a twiddle-free radix-16/32 column kernel into
  a *stack-resident* scratch, then a streamed-twiddle kernel whose strided *sectioned stores*
  produce natural order as a side effect. **No transpose pass exists.** Twiddles are
  streamed linearly from one cursor (~1:1 load:multiply), never recomputed. Byte roofline at
  1024: 4 sweeps × 16 KB + one table stream.
- The mono tier **stops at N=64** — a fact we later reproduced from the inside (see §5.5).

### 2.2 FFTW 3.3.10 source study (local tree)

- No execute-time twiddle recurrence anywhere; tables are built per-entry at plan time in
  long-double with octant range reduction (accuracy-first; they document rejecting a
  standard recurrence because it "loses several decimal places at 16k sizes").
- The transferable idea: **t3-class twiddle-log3** — load only {w¹, w³, w⁹, w²⁷} per
  butterfly row and reconstruct the rest with ≤2-term in-register complex products (7.75×
  twiddle-byte reduction at radix-32). This became our O6.

### 2.3 Decomposition probes (our own runtime)

`k1_t1_decomp.c` timed the twiddled combine (t1) against the *identical* DFT with zero
twiddles (n1) at the real pair shapes:

| R1×me | t1 (ns) | n1 (ns) | twiddle share |
|--:|--:|--:|--:|
| 64×16 | 853 | 640 | 25% |
| 32×32 | 529 | 495 | ~0% |
| 64×64 | 4422 | 3726 | 16% |
| 64×128 | 10 666 | 9066 | 15% |

Verdict: twiddle streaming costs 15–29% of the combine **only at R1=64 shapes**; the
dominant cost everywhere is the combine's compute + spill traffic. This one probe prevented
weeks of misdirected twiddle work and correctly ranked the structural levers first.

---

## 3. The nine optimizations

Each entry: the mechanism (in plain terms), the implementation, and the measured result.
Every step was gated — bit-identical where the arithmetic is unchanged, tolerance-gated vs a
naive O(N²) DFT (deterministic AND random inputs) where it is not.

### O1 — Vectorized column pass (the batch identity) · ~2×

The four-step views the N-point transform as an R1×R2 grid: FFT the columns, apply the
twiddle diagonal, transpose, combine the rows. The old BAILEY2 ran the column FFTs one at a
time — one SIMD lane busy, three idle.

**The identity**: the grid's columns sit in memory *exactly* like a batch of independent
transforms — column c's element j is adjacent to column c+1's element j, the same layout the
batch codelets were built for. So one call to the existing `n1_oop` leaf with `count = R1`
computes **four columns per loop iteration in the four AVX2 lanes**, with plain contiguous
loads (no gathers, no shuffles). At 32×32: 8 vector iterations instead of 32 scalar FFTs.

Zero new codelets — the free-stride 11-argument OOP ABI already expressed it. The stage-2
twiddle table needed no changes either: at K=1, BAILEY2's per-lane table *is* the four-step
diagonal.

**Result**: A/BAILEY2 = 0.45–0.52 at every N, output **bit-identical** to BAILEY2 (same
codelet DAG, same rounding — cross-diff exactly 0.0 on every cell).

### O2 — In-place placement · 10–15% at mid-N

The OOP pipeline used three buffer pairs (input, scratch, output). Passing `dst == src` is
safe by construction (the leaf fully drains the input into scratch before anything writes
it), giving a two-buffer working set — the same shape as MKL's `DFTI_INPLACE` config.
Worth 10–15% at 1024–2048 where the working set straddles L1; nothing at ≤512 (already
resident) or 4096 (L2-bound either way). Also the honest comparison: MKL's numbers are
in-place.

### O3 — Two-pass restructure (UL edges) · 8–23%

Post-O1 the pipeline made three full memory trips: columns → **a dedicated transpose trip**
→ combine. The MKL disassembly showed the target: two trips, transpose smuggled into a trip
that happens anyway.

A transpose is just "read in one order, write in another," so we fused it into an adjacent
pass's memory edge — implemented as native **UnitLeg (UL) edge lattices** in the codelet
emitter (a 4×4 in-register transpose in the load preamble or store postamble):

- **Route 2pa**: leaf writes normally; the t1 *reads* the untransposed layout through a UL
  load edge (`radixR_t1_oop_..._UL_UG` twins).
- **Route 2pb**: the leaf *writes* transposed through a UL store edge
  (`radixR_n1_oop_..._UG_UL` twins); the t1 stays standard.

Crucially the twiddle tables are unchanged in both (relabeling groups↔legs preserves the
diagonal), so both routes gate **bit-identical (0.0)** against the 3-pass path. One full
read+write of the array plus one scratch buffer deleted.

**Result**: −8% (512) to −23% (8192), uniformly positive; the two routes trade wins per cell
(2pa favors fat leaves at small N, 2pb wins mid-N) — a calibrator axis.

### O4 — IL-native boundaries · conversion passes → zero

Interleaved `[re,im,re,im]` is the layout real MKL users bring. Serving them from a split
engine costs two full conversion sweeps — unless the conversion rides the codelet edges:

- `il_in` load lattices deinterleave in-register during the column pass's loads — measured
  **completely free** (hidden under the pass's compute).
- `il_out` store lattices interleave during the final stores; combined with the UL edge this
  gives the **true 2-pass IL route** (`t1 UL-load + il_out store` twins): z→z, two passes,
  zero conversion or transpose sweeps — the full MKL two-pass shape on an interleaved buffer.
- Backward on a z buffer cannot use the split engine's pointer-swap identity, so the swap is
  folded into `_sw` (im,re)-swapped lattices: the *forward* DAG with swapped boundary
  reads/writes computes the unnormalized inverse, output in normal order.

**Result**: 2p-il runs within a few percent of split everywhere and **wins the whole cell at
2048** (3511 vs 3661 split). The historical il_derive.py mechanism (mechanically derived IL
twins with broken tails) was deleted — the emitted twins' tail passes re-enter the same edge
emitters at SSE2/scalar widths, correct for any length by construction.

### O5 — The mono tier at N=64 (`--k1-mono`) · MKL-IL parity

For tiny N the whole four-step fits in registers: one emitted function does
[column DFTs on 4-column chunks → twiddle cmuls against **emit-time rodata constants** →
4×4 register transposes → row DFTs → natural-order stores]. No runtime tables, no call
boundaries, no scratch. Validated against a hand-written 29–30 ns oracle first (M1 gate),
then generalized.

**Result**: emitted mono-64 = **30–31 ns vs MKL-IL 30.2–31.4 — parity with their flagship
kernel**, 1.2× over MKL-split. IL and backward variants via the same O4 lattices.

**The equally valuable negative result**: monos at 128/256 *lose* — 98–104 ns vs our own
two-pass 86 at 128, and 272 vs 175 at 256. A mono has zero pass overhead, so the loss is
pure register economics (8+ radix-16 bodies + N-sized register state on 16 ymm = spill
storm). This empirically confirms MKL's own tier boundary on our codelets: **the mono tier
ends at 64 because 128 stops paying on this register file.**

### O6 — LOG3 twiddles (leg-axis derivation) · 10–18% at specific cells

The engine's celebrated FLAT/T1S/LOG3 per-stage twiddle selection did not initially transfer
to K=1: T1S and the lane-engine LOG3 both require the twiddle constant along the SIMD axis,
and the four-step's twiddle *varies* along it (geometrically inadmissible, not just
unimplemented). But FFTW's t3 mechanism — leg-axis derivation — does transfer, because
products of lane-vectors stay lane-vectors.

Emitted via the *existing* `--log3` flag (the substitution machinery was already in the DAG
layer): **252 → 24 twiddle loads at radix-64**, with the *same* Qr/Qi tables (log3 reads a
sparse subset of the same slots — a pure function-pointer swap on existing plans).

**Result**: wins 10–18% at specific (route, cell) slots — 3p@1024 (1316 vs 1608),
2pa@4096 (7426 vs 8272), 3p@8192 — and loses others, i.e. a true measured-selection member.
It took the 4096 verdict in the median calibration. Tolerance-gated (derived twiddles differ
from loaded ones in the last ulp, exactly like FFTW's t3).

### O7 — Linear twiddle layout (twl) · marginal alone, wins per-cell

Repack the FLAT table in consumption order (one advancing cursor per group-quad, the MKL
pattern) instead of R1−1 parallel strided rows. Same values, same bytes, different order —
gates bit-identical. Measured **marginal on average** (−5% to +13%, pair-dependent: the
scheduler emits twiddle loads in DAG order, so the layout is only block-local), and
therefore banked as a calibrator variant rather than a default — where it proceeded to win
**4 of 8 split cells in-context** (64/128/256/8192). A textbook validation of the project's
measured-selection thesis.

### O8 — Per-cell calibration (four-axis wisdom) · 10–20% + composition

The optimum drifts per cell along four coupled axes: **route** {3p, 2pa, 2pb, twl, log3,
mono, il-variants} × **pair** (R1×R2) × **placement** (oop/in-place) × **layout**
(split/interleaved). Fixed choices leave 10–20% on the table; worse, an early inline pair
sweep reproduced the project's known `cmp_old_new` fixed-order thermal bias (first-tried
candidates run coolest and win spuriously).

The calibrator (`benches/calibrate_k1.c`) fixes this by construction: one process per cell,
per-trial **candidate-order rotation**, best-of-4 — and because two full ladder runs showed
*winners flipping* with machine thermal state, verdicts are taken as the **median of three
separated ladder runs** (`k1_aggregate_wisdom.py`) before persisting. Winners land as
kind-3 lines in the same `oop_wisdom.txt` the OOP engine already uses
(`N 1 3 sp_route R1 R2 il_route iR1 iR2 ns` — one line carries both layout axes because the
layout is an execute-time buffer contract).

### O9 — Stride specialization · −42% at 1024, −22% at 4096

The finale, and the biggest single step. Every codelet address is
`base[b*group_stride + j*leg_stride]` with the strides as runtime parameters — so every one
of ~128 accesses per loop iteration pays a multiply-add address computation, on the *same
execution ports the FMAs need*. The K=1 engine calls each codelet with **fixed,
per-cell-known strides**, so they can be baked as compile-time constants: every address
folds into the instruction's memory operand; the arithmetic ceases to exist.

Why this was invisible before: the historical measurement said 6–10% — at fat K, where
hundreds of loop iterations amortize the overhead. At K=1 the loop runs 8–16 times total and
the whole execute is ~1 µs; the address arithmetic had become first-order.

**Result** (same-run, same-pair, gated bit-identical 0.0): 1024 = **731 ns vs 1252
(−42%) — MKL-split parity**; 4096 = **7145 vs 9109 (−22%)**, stacked on top of the UL edge
and log3 (three independent, individually-gated levers multiplying). Implemented as
per-cell twins via a new `--oop-spec-named` flag (stride tuple in the symbol name so twins
of one radix coexist).

---

## 4. Negative results (paid for once, recorded forever)

- **Mono above N=64**: refuted by measurement (see O5) — matches MKL's own fence.
- **Runtime twiddle recurrence**: rejected before building — needs (R1−1)·2 vector state
  (won't fit registers → becomes stack traffic), breaks bit-exact gating, and both MKL and
  FFTW demonstrably avoid it.
- **twl as a default**: marginal (kept as a per-cell variant, where it wins).
- **Lane-engine T1S/LOG3 at K=1**: geometrically inadmissible (twiddle varies along the SIMD
  axis) — the admissible-method set depends on which axis SIMD runs along vs the twiddle
  index. FLAT / linear-FLAT / leg-axis-LOG3 is the K=1 menu.
- **Inline pair sweeps**: structurally thermal-biased; per-cell choice must come from the
  rotated, multi-run calibrator.
- **Single-run calibration**: winners flip with machine state (±20% absolute swings,
  L2-sensitive routes suffer when hot, register-resident monos don't) → 3-run median +
  machine lockdown for paper numbers.

## 5. Trajectories and the final scoreboard

**N=1024, the running thread** (vs MKL-split ≈ 740–766 ns):

| stage | ns | vs MKL-split |
|---|--:|--:|
| scalar tier (start of day) | 3585 | 0.21× |
| BAILEY2 (old routed answer) | 2442 | 0.30× |
| O1 vectorized column pass | 1248 | 0.60× |
| O2 in-place | 1113 | 0.67× |
| O3 two-pass | 1016 | 0.74× |
| O8 calibrated route/pair | 959 | 0.79× |
| O9 stride-spec | **731** | **~1.02× — parity** |

**End-of-campaign scoreboard** (isolated, cooled; MKL columns same-session):

| N | ours (best, route) | MKL-split | MKL-IL | vs split | vs IL |
|--:|--:|--:|--:|--:|--:|
| 64 | **30** (mono) | 33.7 | 30.2 | 1.12× WIN | **parity** |
| 128 | **72.5** (mono-alt, median) | 128 | 68.6 | 1.77× WIN | 0.95× |
| 256 | **154** (2pa 4×64) | 194 | 136 | 1.26× WIN | 0.89× |
| 512 | **360–378** (2pa/2pb) | 337 | 290 | ~0.93× | 0.79× |
| 1024 | **731** (2pb-spec 32×32) | 740–766 | 750–874 | **~parity** | ~parity |
| 2048 | **3399–3511** (2p-il!) | 2373 | 2067 | 0.70× | 0.60× |
| 4096 | **7145** (2pa-l3-spec 64×64) | 4907 | 3980 | 0.69× | 0.56× |
| 8192 | **21 103** (2pa 64×128) | 10 896 | 8989 | 0.52× | 0.43× |

(2048–8192 have known, scoped levers remaining: spec-twin rollout to those cells and the
composed column pass that removes the monolithic-leaf ceiling and its spill wall.)

**Public API reality check**: `vfft_create(C2C, OOP, howmany=1)` + `vfft_execute` — all
gates green across the ladder, forward+backward × split+interleaved, wisdom path and
heuristic-miss path both. The old public route at K=1 (besides being ~4–5× slower at small
N) turned out to be crashable in the champions flow — the K=1 engine is a correctness fix,
not just a fast path, and per user decision it is the unconditional K=1 route (no
kill-switch).

## 6. Method, in five rules

1. **Disassemble, don't speculate**: every structural move copied a measured property of
   MKL's actual kernels (two passes, sectioned stores, tier boundary, table streaming).
2. **Decompose before optimizing**: the t1-vs-n1 probe priced the twiddle stream before any
   twiddle work; the component timings (leaf/transpose/t1) located each bottleneck.
3. **Gate everything**: bit-identical where arithmetic is unchanged (O1, O3, O7, O9 — exact
   0.0), naive-DFT tolerance with det+rand inputs otherwise; no timing believed before its
   gate.
4. **Measure per cell, honestly**: rotation against thermal order bias, medians against
   machine-state flips, incumbent-unless-margin for wisdom.
5. **Bank negatives**: half of the guidance above comes from things measured *not* to work,
   recorded so no future session re-buys them.

## 7. Remaining work (scoped)

1. Production wiring of the spec routes (wisdom ids + aggregator name-map) and rollout of
   spec twins to every winner shape, then recalibration.
2. The composed column pass — replaces the monolithic leaf with a multi-stage batch plan +
   a permuted transpose absorbing the digit reversal; removes the 128-leaf ceiling and its
   structural 128+128 spill traffic; unlocks N=16 384 (64×256).
3. Machine lockdown for publication-grade numbers.
4. r2c / 2D K=1 analogues; AVX-512 twins of the whole family.

## 8. Reproduction

- Gates: `python build.py --src test/test_k1_fourstep.c --compile` (all routes, det+rand).
- Calibration: `benches/calibrate_k1.exe <N>` ×3 runs → `python
  benches/k1_aggregate_wisdom.py <out> <dump1> <dump2> <dump3>` → wisdom dir.
- Public API: `benches/bench_k1_public.exe <wisdom_dir|->` (needs mingw bin on PATH for the
  runtime DLLs).
- vs MKL: `benches/bench_k1_vs_mkl.c --mkl` (same-process, order-flipped);
  `benches/mkl_probes/` for the disassembly recipe (note the `watch -l` and `--args` traps
  documented in its README).
- Codelet regen: `gen_radix.exe` under WSL (opam 5.2.0, `DUNE_CACHE=disabled`, targeted
  `dune build bin/gen_radix.exe`); all flag combinations are recorded in each emitted file's
  provenance header.
