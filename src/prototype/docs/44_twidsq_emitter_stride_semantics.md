# 44. Twidsq Emitter — Proper OOP Stride Semantics

## What was built (continuation of doc 43)

Doc 43 validated the math layer: `dft_expand_twidsq` produces correct
DAGs that flow through Algsimp/Schedule/Emit unchanged. But the
emitter's address arithmetic still used the legacy `slot * K + k`
addressing — slot-major V-interleaved layout that no natural cascade
produces.

This doc completes the emitter side: address arithmetic now uses the
natural row-major OOP semantics with caller-supplied `is`/`os` row
strides. The codelet can be plugged into a multi-stage cascade where
intermediate buffers have arbitrary row stride (padded for alignment,
different stride between stages, etc.) without copying data.

## Function signature

Before (doc 43 emission):
```c
void radixN_n1_fwd_avx512_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im,
    size_t K)  /* number of V-blocks, slot-major V-interleaved layout */
```

After (this doc):
```c
void radixN_twidsq_dit_fwd_avx512_gen(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    const double *tw_re, const double *tw_im,
    size_t is,    /* input row stride */
    size_t os,    /* output row stride */
    size_t V)     /* number of V-blocks; processed vec_width at a time */
```

## Address arithmetic

The codelet decomposes each linear slot index `s` into `(row, col)
= (s / n, s mod n)`. Both input and output use the same formula with
different row strides:

```
in_re[(s/n) * is + (s%n) * V + v]    for slot s = i*n + k  (Input)
out_re[(s/n) * os + (s%n) * V + v]   for slot s = j*n + i  (Output)
```

The math layer's choice of `Output(j*n + i, _)` encodes the transpose:
output position `(j, i)` corresponds to "row j of output is the
DFT result indexed by column i of the input." When the emitter
decomposes this slot back to `(j, i)`, the address `j*os + i*V + v`
naturally writes to the transposed location in the output buffer.

This is symmetric and clean: same formula on both sides, only the
row stride differs.

## Why row stride matters

In a multi-stage cascade, intermediate buffers are rarely tightly
packed. Stage N's output might be laid out with row stride `is` that
includes padding for alignment, or that's inherited from a larger
2D grid. The previous K-batched emission assumed `is = n*V`
(tightly-packed) implicitly via the `slot * K + k` formula. The new
emission separates `is` from `V*n`, letting the caller pick any row
stride that suits the cascade's needs.

For tightly-packed data, the caller passes `is = n*V`. The
decomposition gives `i*n*V + k*V + v = (i*n + k)*V + v` which is
the old slot-major V-interleaved formula. So packed-layout behavior
is preserved — no regression for callers that don't care about
flexible strides.

## Correctness validation

Built a parameterized test (`correctness.c`) that exercises four
stride configurations for R = 2, 4, 8:

```
=== Tightly-packed layout (is = os = n*V) ===
  R=2 packed                   max err = 1.110e-16   PASS
  R=4 packed                   max err = 4.996e-16   PASS
  R=8 packed                   max err = 6.328e-15   PASS

=== Padded input stride (is = n*V + 8) ===
  R=2 in_padded                max err = 1.110e-16   PASS
  R=4 in_padded                max err = 4.996e-16   PASS
  R=8 in_padded                max err = 6.439e-15   PASS

=== Padded output stride (os = n*V + 8) ===
  R=2 out_padded               max err = 1.110e-16   PASS
  R=4 out_padded               max err = 4.996e-16   PASS
  R=8 out_padded               max err = 6.328e-15   PASS

=== Both padded (is = n*V + 16, os = n*V + 24) ===
  R=2 both_padded              max err = 1.110e-16   PASS
  R=4 both_padded              max err = 4.996e-16   PASS
  R=8 both_padded              max err = 6.439e-15   PASS

All 12 configurations PASS.
```

The test uses **8 independent blocks per SIMD batch** (per_block_in_re[v]
has different random data for each v), so all 8 SIMD lanes are
independently verified — not just lane 0 as in the doc 43 test.

## What changed in the code

- `lib/emit_c.ml`:
  - `render_load`: added `?(twidsq_n = 0)` parameter; when nonzero,
    decomposes slot as `(s/n, s%n)` and emits `(s/n)*is + (s%n)*V + v`
  - `render_node_def`: threads `twidsq_n` to `render_load`
  - `emit_codelet`: 
    - Adds `?(twidsq_n = 0)` parameter
    - Updates `emit_store` to do the same decomposition for outputs
    - All ~11 call sites of `render_node_def` updated to pass `twidsq_n`
  - Function signature already had `is, os, V` from a previous partial
    edit; this doc adds the matching address arithmetic.

- `bin/gen_radix.ml`:
  - Passes `~twidsq_n:n` to `emit_codelet` when `--twidsq` is set
  - Function name pattern adds a new branch:
    `radix{N}_twidsq_{dir}_{sgn}_{isa}_gen{...}` instead of being
    misclassified as `n1` (no-twiddle)

## What's still not done

The emitter changes are complete for **the simple case** (R = 2, 4, 8 with
n ≤ vec_width). Production deployment still needs:

- **DIF direction**: `dft_expand_twidsq` only implements DIT.
  Adding DIF is parallel construction (~30 lines), same structure but
  post-twiddle the DFT outputs instead of pre-twiddling inputs.

- **TP_Log3 twiddle policy**: currently only TP_Flat. For twidsq with
  many distinct twiddles (R=8 has 49, R=16 has 225), log3 could
  meaningfully reduce twiddle table size. Each row's twiddle set is
  `W^{i*1}, W^{i*2}, ..., W^{i*(n-1)}`; the log3 derivation could
  share base twiddles across rows.

- **Spill marker support for large R**: currently no spill plumbing for
  twidsq DAGs. R=16 twidsq has 4087 DAG nodes (doc 43), well into
  spill territory. Need to extend the spill controller to recognize
  twidsq's CT decomposition boundary (the per-row DFT) and emit
  spill markers accordingly. This is a non-trivial extension because
  twidsq has n parallel sub-DFTs, not one — the spill structure is
  different.

- **n > vec_width packing**: for R=16 on AVX-512 (n=16, vec_width=8),
  each "row" of the n×n block is 16 doubles = 2 ZMMs. The current
  address arithmetic `(s/n)*is + (s%n)*V + v` works fine for any n;
  but the SIMD width V = vec_width = 8 still applies. So for R=16 V=8,
  each codelet call processes 8 blocks where each block has 16*16
  doubles. The address `(s/n)*is + (s%n)*V + v` reads the correct
  element. Should work; just hasn't been benchmarked yet.

- **R=1 transpose-only mode**: a degenerate twidsq with n=1 would be
  a pure transpose (no DFT). FFTW has such codelets ("hc2c_inplace").
  Not strictly necessary for our use case but worth noting.

## Where this fits in the OOP roadmap

```
Doc 43 (last session):
  ✓ Math layer: dft_expand_twidsq produces correct DAGs
  ✓ Algsimp/Schedule/Emit pipeline handles them unchanged
  ✓ Correctness validated at R=2,4,8 with K-batched stride

This doc (44):
  ✓ Emitter address arithmetic uses natural row-major OOP layout
  ✓ Caller-supplied is/os row strides
  ✓ Correctness validated at R=2,4,8 with packed AND padded layouts
  ✓ Function names reflect twidsq nature

Next (deferred):
  - DIF variant for the math layer
  - Spill markers for large twidsq
  - n > vec_width benchmarks (R=16 first)
  - Planner integration: how does the planner decide when to use
    twidsq vs separate notw + stride_twiddle_transpose passes?
  - Interleaved-complex layout support (the user mentioned this as
    a possible next session topic)
```

The twidsq codelet now has correct production-quality OOP semantics.
A cascade using stage-1 (notw or t1_dit) followed by stage-2 (twidsq)
can pass intermediate buffers between stages with arbitrary row strides
and get correct results.
