# 44. OOP Twidsq Emitter — Stride Parameterization

## What was built

Doc 43 validated that the math layer for twidsq is correct (CSE/Algsimp/
Scheduler handle the OOP DAGs unchanged), but the emitter still used the
K-batched OOP pattern: `in_re[slot * K + k]`, treating the n² block as a
linear stride-K array. Correct mathematically but unusable for real
multi-stage cascades, where input and output strides differ stage-to-stage.

This doc covers the emitter change to support arbitrary input/output
strides via a new `twidsq` codelet kind.

## API change

A new `--twidsq` codelet path in `emit_codelet` produces this signature:

```c
void radix<N>_<variant>_avx512_gen(
    const double * __restrict__ in_re,
    const double * __restrict__ in_im,
    double       * __restrict__ out_re,
    double       * __restrict__ out_im,
    const double * __restrict__ tw_re,    /* (n-1)² broadcast twiddles */
    const double * __restrict__ tw_im,
    size_t is,    /* input row stride between successive slots */
    size_t os,    /* output row stride */
    size_t V)     /* V-block count; processed vec_width at a time */
{
    for (size_t v = 0; v < V; v += <vec_width>) {
        /* Loads:  in_re[slot * is + v] for slot = i*n + k       */
        /* Stores: out_re[slot * os + v] for slot = j*n + i      */
        /*         (TRANSPOSED — math layer's Output index handles it) */
    }
}
```

vs. the existing OOP path which uses a single `K` parameter as both
the loop bound and the stride.

## Why the transpose is "free"

The math layer (doc 43) builds the DAG with `Output(j*n + i, ...)` for
row i's j-th DFT output. The emitter renders this literally as
`out_re[(j*n + i) * os + v]`. The caller's output buffer therefore
receives data at the transposed positions WITHOUT any in-register
shuffle pipeline. The math layer's index choice IS the transpose.

This is meaningfully different from the FFTW gen_twidsq.ml pattern which
uses in-register transpose intrinsics (`_t8x8` for AVX-512) to vectorize
within the n×n block. Our pattern vectorizes ACROSS V batches instead,
which is simpler to emit and matches the existing K-batched codelet
architecture. The line-filling property of FFTW-style transpose is
deferred for now; the cost is one extra cache transition per cascade
stage at the worst case. Whether this matters in practice depends on
how often cascades run; if it shows up in profiles we can add the
in-register transpose path as a separate twidsq variant.

## Twiddle handling

Twidsq codelets broadcast twiddles via `_mm512_set1_pd(tw_re[slot])`
rather than wide-loading. The reason: twiddle values depend on the
inter-stage (i, k) decomposition, not on which V-batch we're processing,
so they're uniform across V lanes. This matches FFTW's `LDW` convention
and matches what t1s codelets already do for the same reason in a
different context.

In the emitter, this is enforced by `tw_broadcast = t1s || twidsq` in
`render_load` — twidsq codelets use the t1s twiddle path regardless of
the `--t1s` flag's value.

## Correctness validation

Ran 8 (R, is, os) combinations covering:

```
R    is    os    case                          ref_err    lane_err   status
2    8     8     tight is=os=V                 5.55e-17   0          PASS
2    16    24    padded different is/os        5.55e-17   0          PASS
4    8     8     tight is=os=V                 3.33e-16   0          PASS
4    32    16    different in/out padding      3.33e-16   0          PASS
4    128   8     large in pad, tight out       3.33e-16   0          PASS
8    8     8     tight is=os=V                 6.22e-15   0          PASS
8    16    24    different in/out padding      6.22e-15   0          PASS
8    64    8     cascade-style strided in      6.22e-15   0          PASS
```

`ref_err` is the max divergence from a scalar reference; all within FP
precision (~ε·N). `lane_err = 0` across all cases proves the codelet
reads/writes the right addresses for every V lane — since the test
broadcasts each scalar input to all V lanes, any address-arithmetic bug
would show as different lanes producing different outputs.

## Files changed

- `lib/emit_c.ml`:
  - Added `?(twidsq = false)` parameter to `render_load`, `render_node_def`,
    `emit_codelet`
  - When `twidsq=true`: stride is `is` for inputs, `os` for outputs,
    loop variable is `v` not `k`, twiddles always broadcast, function
    signature includes `is, os, V`
  - Assertion: `twidsq && in_place` is rejected at codelet emission time
- `bin/gen_radix.ml`: passes `~twidsq:!twidsq` to `emit_codelet`
- All 10 `render_node_def` callsites updated to propagate `~twidsq`

The twidsq parameter defaults to `false` everywhere, so existing codelet
generation is unaffected. Prime correctness 56/56 PASS confirms.

## What this enables

The OOP twidsq codelets can now be plugged into a multi-stage cascade:

- Stage N produces output at stride `os_N`
- Stage N+1 (a twidsq) reads at stride `is_{N+1} = os_N`
- Stage N+1 produces output at stride `os_{N+1}` (which becomes
  `is_{N+2}` for the next stage)
- Each stage's strides match the surrounding cascade's data layout

For the planner's purpose: the same `gen_radix.exe <R> --twidsq --emit-c`
command now produces a codelet with the API a real planner needs. No
more "this only works at K=1" caveat from doc 43.

## What's NOT in this change

- **In-register transpose pattern** (FFTW-style `_t8x8`). The current
  emission vectorizes across V batches; for cases where line-filling
  cache behavior matters, we'd add a parallel variant that uses
  `_t8x8`-style shuffles. Deferred.
- **DIF direction**. The math layer's `dft_expand_twidsq` only
  implements DIT; calling with `--dif` raises `failwith`. Adding DIF
  is mechanical (mirror the existing DIF construction in
  `dft_expand_twiddled`) once we have a real use case.
- **Spill markers for large R**. R=8 twidsq has 773 DAG nodes — not
  yet at spill threshold, so this works without spill. Larger R (16+)
  would need `dft_expand_twidsq_spill` analogous to
  `dft_expand_twiddled_spill`. Deferred until we want twidsq codelets
  at those sizes.
- **Interleaved (AoS) variant**. FFTW's `_c` codelets store complex as
  `[re0, im0, re1, im1, ...]` instead of separate re/im arrays. Useful
  for some cache patterns; flagged for the next session.
- **Function naming**. The variant classifier still labels twidsq
  codelets as `n1` (no-twiddle) — cosmetic, but the function names
  should distinguish twidsq from other variants. The current name
  pattern `radix<N>_n1_fwd_avx512_gen` doesn't communicate that this
  is a twidsq codelet at all. Easy fix; deferred.

## Stack-op baseline for future spill work

R=8 twidsq compiled with gcc-11 + -flive-range-shrinkage and the new
OOP signature — record the baseline now so when we add spill markers
later we can quantify the win:

```
R=8 twidsq, V=8, gcc-11 + shrinkage:
  Compiled size: ~30KB asm  (vs ~9KB for R=8 t1_dit_fwd)
  ZMM register usage: high (DAG has 226 loads + 773 nodes)
  No explicit spill — relying on GCC's auto-allocator
```

R=8 t1 codelets fit comfortably without spill (peak ~12 live values),
but R=8 twidsq has roughly 8× the work (R parallel DFTs in one codelet)
so the live-set is larger. We don't have asm-level stack-op counts yet
for the twidsq variant; that's the diagnostic to run once we want to
add spill support.
