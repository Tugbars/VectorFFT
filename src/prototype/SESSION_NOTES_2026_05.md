# Session notes — 2026-05 (FFTW comparison + r2c foundation)

This session built on the doc 44 state and added the following.

## New library modules

- **`lib/oracle.ml`** (new, ~150 lines)
  Randomized-CSE diagnostic. Detects algebraically-equivalent-but-
  structurally-distinct nodes in the post-algsimp DAG by evaluating
  with random FP values and bucketing by relative-precision hash.
  Exposed via `gen_radix.exe N --twiddled --in-place --oracle-diag`.
  Diagnostic only, not in production paths.
  See doc 47.

- **`lib/dft_r2c.ml`** (new, ~130 lines)
  Math layer for real-to-complex DFT. Separate module from `dft.ml`
  per FFTW convention (one module per transform type). Currently
  has `dft_r2c_direct` (forward via pair-pack + Hermitian-extraction
  butterfly, all fused into one DAG) and `dft_expand_r2c`.
  Exposed via `gen_radix.exe N --r2c`. See doc 49.

## Library extensions (existing files)

- **`lib/algsimp.ml`** — added `factor_common_terms` and
  `count_factor_opportunities`. Research artifacts (never called by
  production paths) for measuring non-constant factoring
  opportunities. Result: zero opportunities found at any radix; see
  doc 48.

## Bench infrastructure

- **`bench/regression/regression_bench_avx2.c`** (new) — AVX2 sibling
  of the existing AVX-512 regression bench. Same R={16, 25, 32, 64}
  hand-vs-OCaml comparison structure.
- **`bench/regression/build_and_run.sh`** — now drives both ISAs by
  default. `ISA=avx512`, `ISA=avx2`, or `ISA=both` (default). Default
  `-march` updated from `skylake-avx512` to `icelake-server` for
  production ICX target.

## New tests

- **`test/r2c/{verify.c, sweep.c, README.md}`** — R=16 r2c forward
  correctness and K-sweep harnesses. Verifies against direct-DFT
  reference, reports ns/call across K = 8..1024.

## New docs

- **doc 46** — what FFTW's algsimp does that ours doesn't
  (comparison study; identifies Oracle, generalized collect, and
  deepCollect as the FFTW features we lacked)
- **doc 47** — Oracle experiment results (negative: zero missed CSE)
- **doc 48** — Non-constant factoring experiment (negative: zero
  opportunities)
- **doc 49** — R=16 r2c first working codelet, end-to-end verified

## Key findings

1. **Our structural CSE is algebraically complete.** Oracle finds
   zero missed CSE across R=5 through R=256 and all variants tested.
   Different mechanism than FFTW (structural normalization vs
   randomized hashing) but same end state.

2. **Non-constant factoring has zero opportunities** in our DAGs.
   Our math-layer construction never creates the redundant-multiply
   patterns FFTW's generalized collect targets.

3. **R=16 r2c works on first compile**, ~half the ops of R=16 c2c as
   predicted by Hermitian symmetry. FMA gap visible (0 FMAs vs c2c's
   33) — deferred fix in either math layer or emit_c.

4. **AVX2 regression bench** confirms OCaml at parity or better than
   hand across R={16, 25, 32, 64}. R=25 algsimp gap from AVX-512 does
   NOT reproduce on AVX2.

## Decisions taken

- **Drop Oracle implementation.** Zero opportunities found across
  R=5..256. Diagnostic stays in tree as research artifact.
- **Drop generalized collect.** Same reason.
- **Drop deepCollectM by inference.** Two consecutive negative
  results from doc 46's top candidates argue strongly that the third
  would also be zero.
- **CSE chapter closed.** Real perf gaps live below the algsimp
  layer (scheduler choice on R=25, executor overhead, FMA gap in
  r2c). No further investment in algsimp passes.
- **r2c built in OCaml.** Decision per session discussion: unified
  generator long-term, OCaml's symbolic math layer handles the
  Hermitian-symmetry reasoning naturally.

## Open work going forward

- R=32, R=64 r2c forward (mechanical extension of doc 49)
- R2C backward (c2r) — symmetric to forward, doc 49 outlines path
- FMA gap in r2c — 20-30% throughput improvement available
- Cascade boundary codelets (`t1_r2c_first_R` / `t1_r2c_last_R`
  family) per v1.1 roadmap
- Bench r2c against the existing pack→c2c→butterfly executor in r2c.h
