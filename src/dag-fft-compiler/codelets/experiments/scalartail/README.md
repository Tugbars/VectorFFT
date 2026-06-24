# Arbitrary-K scalar-tail experiment (2026-06-24)

Hand-edited codelets that handle **K not a multiple of the SIMD width** (odd K=7,
K=31, …) by a **bulk full-vector loop + a remainder tail**, instead of the
production codelets' `for (k=0; k<me; k+=VW)` which overruns at odd K.

These are **experiment artifacts, not built by default** — `experiments/` is not
in `build.py`'s `dag_codelet_srcs()` glob, so they don't compile/link and don't
clash with the production symbols. The production `codelets/inplace/avx2/`
versions are the unmodified generated ones.

## Files
- `r4_n1_fwd.c` — radix-4 n1: bulk + **scalar** remainder loop (default) **or**
  one **forward masked** vector pass (`-DVFFT_TAIL_MASKED`).
- `r4_t1s_dit_fwd.c` — radix-4 t1s (broadcast tw): bulk + scalar remainder loop.
- `r8_t1s_dit_fwd.c` — radix-8 t1s, **blocked two-pass**: bulk (unchanged) + tail
  via an **offset-call** to the scalar codelet below (t1s broadcast tw ⇒ no
  me-stride, so `radix8_t1s_dit_fwd_scalar(rio+k, …, ios, me-k)` is correct).
- `r8_t1s_dit_fwd_scalar.c` — the generated `--isa scalar` radix-8 t1s, used as
  the r8 tail helper (verified bit-exact vs avx2 in tail_validate).

Tail mechanics: the remainder tail must live **inside** the codelet (same `me`),
because the twiddle leg-stride is `me`. Even-K is byte-identical to the original
(the tail block never fires). Scalar bodies were lifted verbatim from the
`--isa scalar` generated codelets (which compile to SSE-scalar `vmovsd`/`vaddsd`,
0 x87 — the 1-wide rung of the SIMD-width cascade).

## How to reproduce
1. Copy these into `codelets/inplace/avx2/` (overwriting the production ones).
2. Rebuild the codelet lib + the bench: `python build.py --src benches/bench_oddk_tail.c --mkl --compile`.
3. Run via the setvars wrapper (one cell per process; `bench_oddk_tail.exe <K> <flip 0|1> <cool_ms>`).
   Driver loops K × both flips as fresh processes (canonical measure_ab method).

## Findings
- **Correctness:** bit-exact (`0.00e+00`) at every odd K for the all-radix-4 plan
  `N=1024 [4,4,4,4,4]` T1S (vs a padded-even reference; lanes are independent).
- **Margin (canonical method: per-cell process isolation, best-of-5 min,
  cachebust+cool between engines, order-flip averaged):** odd-K beats MKL
  **~1.6–1.8×**, even-K ~1.7–1.9×. flip0≈flip1 confirms no order bias.
- **The scalar tail is correct but NOT optimal.** Margin tracks the scalar-lane
  fraction `rem/K`: at rem=1 we're at the even-K margin (~1.8×, on par with our
  peak); as rem grows the margin erodes (K=31 rem=3 → 1.61×; K=15 rem=3/15 →
  1.29×) — i.e. **MKL's remainder method is more efficient than our scalar tail**
  (its per-transform cost stays ~flat across K; ours rises ~`rem`× per scalar
  lane). The fix is a **vector-efficient remainder** (one masked pass for the
  whole remainder, not `rem` scalar passes) — the next experiment.
- **Open bug:** the blocked two-pass r8 stage corrupts *bulk* lanes at odd K
  (~4e2), **independent of the tail** (identical error with the hand-splice AND
  the verified scalar offset-call; bulk is byte-identical to the original; r4-only
  plans are clean). Points at the executor/twiddle/seam handling of blocked
  codelets at odd K, not the codelet tail.

The production path forward is the generator change (emit the tail per codelet),
not these hand-edits. See `docs/roadmap/arbitrary_k_vectorization.md`.
