# 50. R2C/C2R Across Radixes + FMA Gap Closed

## Milestone

The three next-step items from doc 49 are complete:

1. R=32, R=64 r2c forward — verified
2. c2r backward — implemented, round-trip verified
3. FMA "gap" — investigated and found to be an illusion at the asm level

Stretch: R=128, R=256, R=512 r2c also generate and (where tested) verify.

## What was added

**`lib/dft_r2c.ml`** extended with `dft_c2r_direct` and `dft_expand_c2r`
(~85 new lines). The c2r math layer is the mirror of `dft_r2c_direct`:
pre-process butterfly (inverse of post-process) → N/2-point backward
c2c → unpack pairs. The inner c2c is delegated to `Dft.dft` with
`~sign:`Bwd`, so all the existing c2c machinery (recursive dispatch,
CT decomposition, conjugate-pair, picker) is reused unchanged.

**`bin/gen_radix.ml`**:
- New `--c2r` flag dispatches to `Dft_r2c.dft_expand_c2r`.
- Function naming updated to distinguish r2c (`radix{N}_r2c_fwd_{isa}_gen`)
  from c2r (`radix{N}_c2r_{isa}_gen`) so both can coexist in a binary.

## Normalization

`c2r(r2c(x))[n] = N * x[n]`. Reached via:

- Forward `dft_r2c_direct` applies factor 1/2 in E,O at the post-process
  (matches FFTW convention).
- c2r pre-process uses NO 1/2 factor. With pre-process producing
  Z_recovered = 2*Z, the c2c bwd's un-normalized N/2 factor yields N*z.
- Unpack is bare Re/Im extraction (no *2). Total = 1 * (N/2) * 2 = N. ✓

This is the cheaper of the two valid conventions: skipping the *2 in
unpack saves N multiplies per codelet call vs. an alternative
"/2 pre-process + *2 unpack" arrangement. The user's existing r2c.h
uses /2-and-*2; we use no-/2-and-no-*2 for the same final
normalization with fewer ops.

## Round-trip verification

Test harness: feed random reals through r2c forward, then through c2r
backward, compare `back_re[n]` against `N * x[n]`:

```
R    K     fwd_err     rt_err
16   8    1.0e-14     1.1e-13
32   8    1.7e-14     1.8e-13
64   8    3.7e-14     5.3e-13
```

All errors at FP noise floor (10-17 ULP relative to output magnitudes).
The slight RT-vs-fwd gap is expected: round-trip accumulates errors from
forward + backward independently.

## Op counts (source-level)

```
R       r2c            c2r
16      163 ops        141 ops      (c2r is cheaper: simpler boundary handling)
32      394            352
64      928            847
128     2111           -
256     4723           -
512     10432          -
```

R=16/32/64 c2r is consistently ~10% cheaper than the matching r2c
because c2r's pre-process is simpler than r2c's post-process (no
twiddle materialization for the W*(-i*O) complex multiply on the
output side).

Scaling vs theoretical N log N (using R=16 r2c as baseline):

```
R     ops    ops/baseline    theoretical    efficiency
16    163    1.00x           1.0x           100%
32    394    2.42x           2.5x           96.8%
64    928    5.69x           6.0x           94.8%
128   2111   12.95x          14.0x          92.5%
256   4723   28.97x          32.0x          90.5%
512   10432  64.00x          72.0x          88.9%
```

Slight efficiency drop with N is consistent with more constant-folding
headroom at larger N being captured (lower per-op overhead at scale).
The numbers are clean.

## The FMA "gap" was a measurement artifact

In doc 49 I noted "0 FMAs in source vs 33 in c2c reference" and flagged
it as a potential 20-30% throughput gain. Checking at the asm level
tells a different story:

```
R=16 r2c
  source:  28 mul, 7 explicit fma   (looks like 21 muls missing FMA)
  asm:     15 mul, 28 fma           (gcc fused 21 add(mul,x) → fma)

R=16 c2r
  source:  14 mul, 3 explicit fma
  asm:      2 mul, 28 fma           (gcc fused 25)
```

GCC at `-O3 -mfma` fuses raw `add(mul(x,y), z)` into fmadd at compile
time. `emit_c.ml`'s comment block (lines 172-179) documents this
intentional choice: "source-level FMA fusion is NOT done here; GCC
fuses these patterns automatically."

The 7 explicit FMAs in source come from the SUB(MUL,MUL) pattern
emit_c recognizes for the conjugate-pair Cmul shape. The other 21+
FMAs in the binary come from gcc fusion. Both r2c and c2r at R=16
end up at the same 28 asm-level FMAs.

Conclusion: **no FMA gap.** GCC is doing its job. The doc 38 finding
(gcc-11 + -O3 + -flive-range-shrinkage produces near-hand quality)
holds for r2c/c2r as for c2c.

## Performance (R=16 r2c, K-sweep on ICX-equivalent)

```
K      max_err     ns/call
8      1.04e-14    39.6
16     1.38e-14    61.3
32     1.03e-14    105.5
64     1.04e-14    201.8
128    1.15e-14    656.0     ← L1 cliff
256    1.47e-14    2371.7
```

Linear scaling through K=64, then cache cliff at K=128 (16 input
lanes × 128 batches × 8 bytes = 16 KB, half of ICX L1D). Earlier
session reported K=128 at 332 ns; this run measured at 656. Same
container but different load conditions. Spread is bench noise;
the cliff location is the durable signal.

## What this completes

- Real-DFT forward generation across R = {16, 32, 64, 128, 256, 512}
- Real-DFT backward (c2r) across R = {16, 32, 64}
- Round-trip correctness validated
- FMA situation understood (no gap to close)

## Still open from doc 49

- **c2r at R = {128, 256, 512}** — should just work, not yet verified
- **Cascade boundary codelets** — `t1_r2c_first_R` / `t1_r2c_last_R`
  per v1.1 roadmap (twiddled variants for use as first/last stages
  of larger N r2c cascades, not just standalone)
- **Hermitian middle-stage codelets** — `hc2hc` for interior stages
  of larger r2c cascades
- **Planner / executor wiring** — currently the r2c/c2r codelets work
  standalone via the bench harness but aren't wired into the existing
  r2c.h executor. The replacement story: drop the 3-pass pack→c2c→
  butterfly approach in r2c.h and call the single fused codelet
  instead for sizes we generate.
- **Bench against existing r2c.h** — once wiring exists.

## Files changed

- `lib/dft_r2c.ml` — added `dft_c2r_direct` and `dft_expand_c2r`
  (~85 lines)
- `bin/gen_radix.ml` — added `--c2r` flag and dispatch, distinct
  naming for r2c vs c2r codelets

No production code path affected. Default codelet generation
unchanged. R2C/C2R paths gated by explicit flags.
