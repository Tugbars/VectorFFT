# 49. R2C R=16 — First Fused Codelet Working

## Milestone

First OCaml-generated r2c codelet, end-to-end. R=16 forward, all 9
output bins correct against direct-DFT reference, ~half the
arithmetic of c2c R=16 as predicted from Hermitian symmetry.

## What was added

**`lib/dft_r2c.ml`** (new module): math layer for real-to-complex and
(eventually) complex-to-real DFTs. Separate from `dft.ml` because r2c
is a fundamentally different transform type with its own decomposition
strategy. Mirrors FFTW's separation between `gen_dft.ml`, `gen_r2cf.ml`,
`gen_r2cb.ml`, etc. — one module per transform type.

The new module currently has:

- `dft_r2c_direct ?sign n input_re`: math layer for real-to-complex
  DFT forward. Takes only a real input source (no `input_im`), returns
  `(expr array * expr array)` of length N/2+1. Internally does
  pair-pack → N/2-point c2c → Hermitian-extraction butterfly, all
  fused into one DAG. The c2c sub-DFT calls `Dft.dft` so all existing
  CT decomposition, conjugate-pair, and recursive dispatch machinery
  is reused unchanged.

- `dft_expand_r2c ?sign n`: assignment-list wrapper. Outputs at slots
  `[0..N/2]`; slots `[N/2+1..N-1]` are not written (Hermitian
  conjugates are caller's responsibility).

As `dft_r2c.ml` grows it will host: `dft_c2r_direct` (backward),
`dft_r2c_first` / `dft_r2c_last` (cascade boundary codelets per v1.1
roadmap), and `dft_hc2hc` (middle stages, Hermitian-packed in & out).

**`lib/dune`**: register `dft_r2c` as a library module. Documentation
header expanded to describe the math-layer / pipeline-layer module
split.

**`bin/gen_radix.ml`**: `--r2c` flag dispatches to
`Vfft_v2.Dft_r2c.dft_expand_r2c`. Calling convention reuses the c2c
codelet ABI for now (`in_re, in_im, out_re, out_im, tw_re, tw_im, K`)
— `in_im`, `tw_re`, `tw_im` are unused at the codelet body level;
gcc optimizes the parameter loads away.

## Verification

Round-trip vs direct-DFT reference (8 batches, K sweep):

```
K      max_err     ns/call
8      1.04e-14    23.2
16     1.38e-14    43.4
32     1.03e-14    83.0
64     1.04e-14    165.7
128    1.15e-14    332.3
256    1.47e-14    1474.9    (L1 cliff)
512    1.29e-14    2941.1
1024   1.42e-14    6977.4
```

All errors at FP noise floor (5-6 ULP relative to output magnitudes
~1.0). Linear scaling through K=128 (cache-resident regime), then
the expected cliff at K=256 where the working set crosses L1 (16
input lanes × 256 × 8 bytes = 32 KB, matches ICX L1D).

## Codelet quality

R=16 r2c forward: **163 SIMD ops**, 16 loads + 18 stores, 113 arithmetic.

For comparison, R=16 c2c t1 twiddled in-place: **341 SIMD ops**, 94 loads
+ 64 stores, 33 FMAs.

The r2c codelet has roughly half the work of c2c, matching the
theoretical 2× speedup from Hermitian symmetry. The reduction comes
from:
- No imaginary input loads (16 reals vs 16 complex = 32 reals)
- Only N/2+1 output bins computed (9 vs 16)
- No runtime twiddle multiplies — twiddle constants are baked in as
  `Const` at math-layer time (this is the "n1" variant; no t1 yet)

## The FMA gap (separate followup)

**Zero FMAs** in the generated r2c codelet, vs 33 FMAs in the c2c
reference. The post-process butterfly produces textbook FMA-eligible
expressions:

```
x_re = e_re + (wr * o_im + wi * o_re)
x_im = e_im + (wi * o_im - wr * o_re)
```

But `emit_c`'s FMA recognition doesn't catch them. The c2c path gets
FMAs through `Cmul` opaque atoms (NK_CmulRe / NK_CmulIm) which have
dedicated FMA emit logic. The r2c path constructs the same arithmetic
via raw Mul + Add and falls through the recognizer.

Two paths forward:
1. **Math-layer side**: rewrite `dft_r2c_direct`'s butterfly section
   to construct `Cmul` atoms for the complex multiply, mirroring how
   `dft_expand_twiddled` produces `Cmul` for input pre-twiddles. The
   complex multiply `W * (-i * O)` is exactly the pattern.
2. **Emit-c side**: generalize the `Add(Mul, x)` pattern recognizer in
   `emit_c.ml` to fire on the raw arithmetic, not just `Cmul`-derived
   nodes.

(1) is more in-the-spirit-of-the-architecture; (2) is more
universally beneficial. Both are independent of the r2c milestone
here. Filing as deferred work.

## Next steps for r2c

In order:

1. **R=32, R=64 r2c forward** — extend coverage. The math layer
   already handles these automatically (recursive `dft` dispatch will
   CT-decompose N/2 = 16 and 32). Should "just work" with the same
   code path. Verify ops scale ~1.7x and ~3.8x respectively.

2. **r2c backward (c2r)** — symmetric algorithm:
   pre-process butterfly (reverse-direction Hermitian extraction)
   → N/2-point c2c backward → unpack pairs. `dft_c2r_direct` mirrors
   `dft_r2c_direct`. Round-trip property `c2r(r2c(x)) == N*x` is the
   natural correctness test.

3. **FMA gap** — fix via path (1) or (2) above. Probably 20-30%
   throughput improvement on the n1 codelet.

4. **t1 (twiddled, in-place) variant** — first-stage and last-stage
   r2c codelets for cascade composition at larger N. This is the
   v1.1 roadmap's `t1_r2c_first_R` / `t1_r2c_last_R` family. Math
   layer extension: don't bake the twiddles in as Const; consume them
   from a `tw_re/tw_im` array at runtime, like the existing c2c t1
   codelets.

5. **Bench against the user's existing r2c.h** — once t1 codelets
   exist and can be wired into a planner, compare end-to-end r2c
   throughput at N=64, 128, 256, 512, 1024 against the current
   pack→c2c→butterfly executor.

## Files changed

- `lib/dft_r2c.ml` — new module (~130 lines): `dft_r2c_direct` and
  `dft_expand_r2c`
- `lib/dune` — register `dft_r2c` module, expanded header comment
- `lib/dft.ml` — unchanged (r2c logic moved out)
- `bin/gen_radix.ml` — added `--r2c` flag and dispatch (3 small edits)

No production code path affected. Default codelet generation
unchanged. R2C path is gated by explicit `--r2c` flag.
