# Session notes — 2026-05 (FFTW comparison, r2c family, cascade Stage A, bounds analysis)

This session built progressively on the doc 44 baseline. Final state
adds c2r backward, head-to-head bench vs 3-pass, the first cascade
boundary codelet, an op-count diagnosis against published bounds, and
a final cleanup pass that removed dead experimental code.

## Library modules

- **`lib/dft_r2c.ml`** (~285 lines) — math layer for real-valued
  transforms. Contains:
  - `dft_r2c_direct` / `dft_expand_r2c`: monolithic r2c forward
    (doc 49)
  - `dft_c2r_direct` / `dft_expand_c2r`: monolithic c2r backward
    (doc 50)
  - `dft_r2c_first` / `dft_expand_r2c_first`: first-stage cascade
    codelet, pack-fused into first DIT pass (doc 52, Stage A)
  All three delegate to `Dft.dft` for the c2c sub-DFT.

## Library extensions and cleanup (existing files)

- **`lib/algsimp.ml`** — now 1855 lines (was 2126). The experimental
  `factor_common_terms` rewriter and `count_factor_opportunities`
  diagnostic (~270 lines) were removed after doc 48's negative
  result. The remaining `factor_terms` references in the file are
  local inner functions inside the legitimate `factor_common_muls`
  pass — name collision, unrelated.
- **`lib/dune`** — `oracle` removed from the modules list.
- **Deleted: `lib/oracle.ml`** (was 207 lines). The randomized-CSE
  diagnostic from doc 47 found zero missed CSE at every radix and
  was never wired into production. Doc 47 retained as the
  permanent finding record.

## Bench infrastructure

- **`bench/r2c_mono/bench_r128.c`** (new) — head-to-head bench at
  N=128 r2c forward comparing the monolithic R=128 codelet to a
  faithful synthetic mirror of `r2c.h`'s three-pass structure
  (pack + R=64 c2c codelet + vectorized Hermitian-extraction
  butterfly). Reports ns/call across K = {8..1024}. Result:
  monolithic wins 1.3-3.3× at every K (doc 51).

## New tests

- **`test/r2c/round_trip.c`** — `c2r(r2c(x)) == N*x` property test.
  Verifies normalization across R = {16, 32, 64}.
- **`test/r2c/verify_r2c.c`** — parametric forward verify with
  K-sweep + timing for any R.
- **`test/r2c/verify_r2c_first.c`** — isolation correctness test for
  the cascade first-stage codelet (any R).
- **`test/r2c/cascade_r16_test.c`** — trivial-cascade end-to-end test.
  At N=16 where N/2=8 fits one sub-DFT, validates
  `(r2c_first_8 → manual butterfly)` matches the monolithic R=16
  r2c codelet to FP precision (5e-15).

## New docs

- **doc 46** — what FFTW's algsimp does that ours doesn't (motivation
  for the experiments that follow)
- **doc 47** — Oracle experiment, negative result, implementation
  removed in cleanup
- **doc 48** — Non-constant factoring experiment, negative result,
  implementation removed in cleanup
- **doc 49** — R=16 r2c first working codelet, end-to-end verified
- **doc 50** — R2C/C2R family complete at R={16, 32, 64, 128, 256,
  512}, round-trip verified, FMA gap explained as compile-time
  fusion artifact
- **doc 51** — N=128 monolithic codelet beats 3-pass mirror 1.3-3.3×
  across K = {8..1024}; architectural premise of fused codelets
  confirmed
- **doc 52** — Stage A: r2c first-stage cascade codelet, isolation
  and trivial-cascade correctness verified at R = {8, 16, 32, 64}
- **doc 53** — Op-count comparison vs Yavne, Johnson-Frigo/LVB, and
  Sorensen split-radix r2c bounds. C2C is at the Yavne bound
  (within 1-3%). R2C is 25-91% over the Sorensen bound, gap
  shrinks with N; identified as the structural cost of monolithic
  post-process vs Hermitian-preserving cascade.

## Key findings

1. **CSE work is complete.** Oracle experiment (doc 47) found zero
   missed CSE across all radixes. Non-constant factoring (doc 48)
   found zero opportunities. Our structural CSE is algebraically
   complete.

2. **C2C is at the published lower bound.** Doc 53 measures our
   c2c codelets at 1-3% over Yavne (1968) split-radix, 3-4% over
   the modern Johnson-Frigo/Lundy-Van Buskirk bound. No abstract-op
   wins remain; future c2c gains must come from FMA-fusion-shape and
   scheduling (both already at parity with hand per doc 38).

3. **R2C cascade architecture validated.** Monolithic R=128 codelet
   beats faithful 3-pass mirror by 1.3-3.3× at every K tested.
   Architectural premise of fused codelets confirmed; cascade
   boundary work is justified.

4. **R2C has one known-better algorithm.** 25-32% op-count reduction
   available at production sizes via FFTW-style hc2c
   (Hermitian-preserving cascade, Sorensen 1987). Doc 53 quantifies
   the prize.

5. **r2c_first cascade boundary codelet works.** Math layer is
   trivial (4 lines: `Dft.dft` with pair-packed indexing).
   Verified in isolation across R = {8, 16, 32, 64} and in
   trivial-cascade end-to-end at N=16.

## Open work (in priority order)

1. **Stage B** — Build a 2-stage cascade harness at N=128 using the
   pieces we have (`radix8_r2c_first` × 8 + inter-stage twiddles +
   existing `radix8_t1_dit` × 8 + Hermitian-extraction butterfly).
   Bench against monolithic R=128 and 3-pass mirror. Isolates how
   much of the mono-vs-3pass win comes from pack-fusion alone.

2. **Stage C** — FFTW-style hc2hc + hc2c codelets, Hermitian-preserving
   cascade. Captures the 25-32% op-count reduction quantified in
   doc 53. Math layer needs new functions for Hermitian-packed
   intermediate data (separate from `Dft.dft`).

3. **Planner integration** — Replace `r2c.h`'s 3-pass approach with
   cascade composition using the new codelets. Picker entries for
   N ∈ {128, 256, 512, 1024, 2048, 4096}.

4. **Real-hardware bench on ICX** — Container numbers from this
   session need confirmation on real ICX. Expected similar or
   better ratios.

## Deferred

- **JF/LVB conjugate-pair twiddle reorganization** for the final ~1%
  c2c op-count savings to JF/LVB. Math-layer change in `dft.ml`
  (not algsimp). Decided not worth the ~week of work given the
  small realized win at our production sizes. Path is clear if
  revisited later (v1.2+).

- **`lib/split_radix.ml`** kept in tree (not wired in) as future
  hedge against CPU architecture changes that might favor
  multiplication-light algorithms.

## Verified state at end of session

- `dune build` clean
- Prime correctness: 56/56 PASS
- All production codelet types generate: c2c plain, c2c twiddled,
  r2c, c2r, r2c_first
- Algorithmic op count: at Yavne bound for c2c, 25-91% over
  Sorensen for r2c (known-better algorithm identified)
