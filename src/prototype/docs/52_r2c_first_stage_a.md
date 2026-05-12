# 52. Stage A: r2c First-Stage Cascade Codelet

## Milestone

Built and verified `dft_r2c_first` — the first cascade boundary codelet
for r2c. This is the easy half of the cascade work: pack fusion into
the first stage. The harder half (Hermitian-extraction butterfly
fusion into the last stage) is deferred to Stage C.

## What was added

**`lib/dft_r2c.ml`** extended with:

- `dft_r2c_first ?sign r input_re`: math layer for a no-twiddle
  first-stage codelet. Takes a real input source, treats it as
  pair-packed complex (`z[k] = input_re(2k) + i * input_re(2k+1)`),
  delegates to `Dft.dft r` for the R-point complex DFT.

  Implementation is trivial — the pack is just an indexing
  reinterpretation at math-layer time. The c2c sub-DFT goes through
  the existing recursive dispatch (CT decomposition, direct,
  conjugate-pair) unchanged.

- `dft_expand_r2c_first ?sign r`: assignment-list wrapper. Inputs at
  slots `Input(j, true)` for `j = 0..2R-1` (2R reals total). Outputs at
  `Output(k, true)` and `Output(k, false)` for `k = 0..R-1` (R complex).

**`bin/gen_radix.ml`**:
- New `--r2c-first` flag dispatches to `Dft_r2c.dft_expand_r2c_first`
- Function naming: `radix{R}_r2c_first_{sgn}_{isa}_gen`
- The `{R}` is the sub-DFT radix (not total transform size)

## Verification

### Isolation test (verify_r2c_first.c)

Tests the codelet in isolation: feed 2R random reals, check output
against direct DFT of pair-packed input.

```
R=8   K=8  max abs err: 4.8e-15  PASS
R=16  K=8  max abs err: 1.4e-14  PASS
R=32  K=8  max abs err: 4.9e-14  PASS
R=64  K=8  max abs err: 1.3e-13  PASS
```

Errors grow with R (more FP operations accumulating) but all at noise
floor.

### Trivial-cascade test (cascade_r16_test.c)

Validates that the first-stage codelet is mathematically correct **for
cascade use**, not just in isolation. At N=16 where N/2=8 fits in one
sub-DFT, a "1-stage cascade" is `r2c_first_8 + butterfly` — should
produce identical output to the monolithic R=16 r2c codelet.

```
Monolithic vs (r2c_first_8 + butterfly):
  max diff = 5.0e-15 (FP noise, 4-5 ULP)
  All k bins match to FP precision across 8 batches
```

This is the key validation. The math layer of `r2c_first` is correct
when composed with the Hermitian-extraction butterfly. Scaling to
true multi-stage cascades (N > 16) is a planner-wiring problem, not a
math-layer problem.

## Op profile

```
R=8   first-stage:    89 ops    16 loads, 16 stores,  4 mul,  25 add, 26 sub
R=16  first-stage:   234 ops    32 loads, 32 stores, 23 mul,  69 add, 72 sub
R=32  first-stage:   593 ops    64 loads, 64 stores, 82 mul, 176 add, 188 sub
R=64  first-stage:  1432 ops   128 loads,128 stores,232 mul, 430 add, 464 sub
```

Op counts match plain R-point c2c codelets at the same R. This is
correct — the pair-pack is free at math-layer time (just indexing
reinterpretation), so the first-stage codelet does exactly the same
arithmetic work as a c2c codelet. The win is at the **integration**
level: the first-stage codelet reads 2R reals directly from
contiguous input rather than reading from a pre-packed complex
intermediate buffer (which would require an extra pack pass).

## What this enables

The building blocks for true multi-stage r2c cascades now exist:

1. **First stage**: `radix{R}_r2c_first_{sgn}_{isa}_gen` (this doc) —
   pair-packed real → complex
2. **Middle stages**: existing `radix{R}_t1_dit_{sgn}_{isa}_gen_inplace`
   (c2c twiddled, no new code)
3. **Last stage of inner c2c**: same as middle — regular twiddled c2c
4. **Hermitian-extraction butterfly**: written manually for now
   (vectorized in bench_r128.c style); eventual Stage C work fuses
   this into last-stage codelet via FFTW-style hc2c machinery

For a 2-stage cascade at N=128 (factoring N/2 = 8×8):
- 8 calls of `radix8_r2c_first_fwd_avx512_gen` (different input offsets)
- Inter-stage twiddle multiplication (scratch buffer)
- 8 calls of `radix8_t1_dit_fwd_avx512_gen_inplace` (with stride-8 read pattern)
- Hermitian-extraction butterfly (one pass over full output)

The codelets exist. The planner wiring is what's left.

## Stage B (next session, suggested)

Build a 2-stage cascade harness at N=128 and bench it against:
- Monolithic R=128 r2c codelet (doc 50)
- 3-pass synthetic mirror (doc 51)

This isolates how much of the monolithic-vs-3pass win (1.3-3.3×, doc
51) comes from pack-fusion alone (Stage A) vs needing full butterfly
fusion (Stage C).

Expected behavior:
- If pack-fusion alone captures >80% of the win, Stage C is
  lower-priority decoration.
- If pack-fusion captures <50% of the win, Stage C is urgent (the
  butterfly pass is the bottleneck).

Either result is informative.

## Stage C (later)

FFTW-style hc2hc + hc2c codelets that preserve Hermitian symmetry
throughout the cascade and fuse the butterfly into the last stage.
This is the substantial work — entire new generator paths for
Hermitian-packed intermediate data. Defer until Stage B numbers show
it's worthwhile.

## Files

- `lib/dft_r2c.ml` — added `dft_r2c_first` and `dft_expand_r2c_first`
  (~60 lines)
- `bin/gen_radix.ml` — `--r2c-first` flag, dispatch, distinct naming
- `test/r2c/verify_r2c_first.c` — isolation correctness test (any R)
- `test/r2c/cascade_r16_test.c` — trivial-cascade end-to-end test

No production code path affected. Default codelet generation
unchanged. New path gated by `--r2c-first` flag.
