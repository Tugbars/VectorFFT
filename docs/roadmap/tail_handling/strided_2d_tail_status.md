# Strided / 2D tail status — what actually needs a tail (2026-06-30)

> **Finding (traced end-to-end):** the **production 2D path already handles odd
> dimensions today** — there is no missing "strided tail" there. The Design-C
> `*_strided` codelets are a *separate, unused, single-radix-only* alternative
> (roadmap). The **one real arbitrary-K gap left is the decoupled-stride r2c inner
> (`n1_oop_strided`, the `K%8` gate)** — that's the only path that actually refuses
> odd K.

This corrects an earlier over-statement that the strided codelets needed a tail to
unblock 2D. They don't; 2D works at odd sizes now.

## Two different "strided" things — don't conflate

| | Design-C `*_strided` codelets | `transpose.h` + native (the live 2D path) |
|---|---|---|
| what | one codelet fuses load-rows → in-register transpose → butterfly → inverse → store-rows, no scratch | standalone SIMD transpose to scratch, then the ordinary multi-stage executor, then transpose back |
| wired in? | **NO** — only consumer is a link smoke-test (`build_tuned/benches/benchmarks/gate_strided_registry.c`) | **YES** — `fft2d.h` |
| coverage | **n1 only, single radix ≤64** ([emit_c.ml:1668](../../../src/dag-fft-compiler/generator/lib/emit_c.ml#L1668)); can't do composite N2 | any N1, N2 |

## The live 2D row-FFT call chain (traced)

```
_fft2d_execute_fwd                                          fft2d.h:273
├─ Phase 1 (cols): stride_execute_fwd(plan_col, re, im)     fft2d.h:280
│     └─ N1-pt FFT, K=N2, native vertical — NO transpose
│        (odd N2 ⇒ odd batch K=N2 ⇒ vertical SSE2 tail)
└─ Phase 2 (rows): _fft2d_tiled_mt → _fft2d_tiled_range     fft2d.h:116
      for each tile i (i += B):
        this_B = min(B, N1 - i)                             fft2d.h:126  ← partial last tile
        stride_transpose_pair(re+i*N2,…, N2, B, this_B, N2) fft2d.h:129  GATHER this_B×N2 → N2×this_B
        vfft_proto_execute_fwd(plan_row, sr, si, this_B)    fft2d.h:143  ROW FFT N2-pt, K=this_B (native)
        stride_transpose_pair(sr,…, B, N2, N2, this_B)      fft2d.h:146  SCATTER back

stride_transpose_pair                                       transpose.h:372
└─ _rec_* → _base_A / _base_B                               transpose.h:239 / 267
     • SIMD _t4x4 / _t8x4 / _t8x8 for FULL VW×VW blocks
     • SCALAR edge loops for leftover rows/cols             transpose.h:255-257, 284-286, 314-317
```

**Why odd dimensions already work:**
1. The `*_strided` codelets appear **nowhere** in this chain.
2. `stride_transpose_pair` is a general transpose — SIMD kernels for full blocks **+ scalar edge loops** for the leftover rows/cols ([transpose.h:255](../../../src/core/transforms/fft2d/transpose.h#L255)). Partial tile (`this_B < B`) and odd `N2` are correct; the edge just runs scalar.
3. Both inner FFTs run odd batch through the **vertical SSE2 tail** we already shipped — column pass `K=N2`, row pass `K=this_B`.

⇒ **2D needs no tail work for correctness.** At most, the transpose's scalar edge at odd sizes is a minor perf cost (could later get an in-register/masked edge — *not* a gate).

## The Design-C strided codelet tail (roadmap, if ever wired)

If those codelets are ever wired in (only helps single-radix N2), their tail is the
**in-register row-padding** design (no mask, no copy): bulk `for(b; b+VW<=me; b+=VW)`,
then `if(b<me)` re-emit the transpose preamble with the load index **clamped**
`min(b+r, me-1)` and the postamble store **guarded** `if(b+r<me)`; body reused
verbatim. Tail and pad coincide there (free in-register padding). Slot: the `strided`
branch of `emit_codelet` ([emit_c.ml:1656](../../../src/dag-fft-compiler/generator/lib/emit_c.ml#L1656)), flipping the line-1555 exclusion. **Low priority** — 2D already
works and these codelets are unused.

## `n1_oop_strided` (decoupled-stride r2c inner) — VERIFIED: not a hard tail

The decoupled-stride r2c strategy (real FFT via N/2 complex, the *second* r2c path vs
the rfft cascade) uses a `n1_oop_strided` inner. Its `K%8` gate
([r2c_dispatch.h:216](../../../src/core/transforms/real/r2c_dispatch.h#L216), shared
`_vfft_r2c_build_stride` `if(K%8)return NULL`) is the only place odd K is actually
*refused*. **Verified what it is: a VERTICAL codelet, not a transpose.** Signature
`(in_re, in_im, out_re, out_im, is, os, vl)`, body `in_re[leg*is + v]`, loop over `vl`
([emit_c.ml:2214](../../../src/dag-fft-compiler/generator/lib/emit_c.ml#L2214)) —
identical shape to `r2cf`/`c2r`/`hc`.

It's excluded from `anyk_tail` **by deliberate routing, not difficulty**
([emit_c.ml:1552-1557](../../../src/dag-fft-compiler/generator/lib/emit_c.ml#L1552):
*"stride inner never needs it"* — the executor always passes a VW-multiple `B`, and odd
K is routed to the rfft cascade). So its tail is the **exact SSE2/scalar mechanism
already shipped**. Removing the gate would be:
1. add `!n1_oop_strided` to `real_fft_sig` — tail emits at bound `vl`, var `v`,
   byte-identical to the other real families (**one line**);
2. make the decoupled executor pass the real odd `vl`, not a padded `B`;
3. ensure the Hermitian fold handles odd K;
4. relax the `K%8` gate.

**LOW VALUE — likely don't.** Per [[r2c-st-loses-mt-wins]]: r2c ST loses to MKL by
design and odd-K r2c is *uncalibratable*, so this path ST-loses regardless; forcing odd
K onto the rfft cascade is the *better, calibratable* routing. The gate is partly that
deliberate choice. Completeness item, not a win.

## Bottom line: no HARD tail remains

Every family is either already handled — in-place / OOP / rfft / c2r / trig via the
SSE2 tail; **2D via the transpose's scalar edges + the vertical tail** — or a vertical
codelet with the existing tail (`n1_oop_strided`, one-line enable). The coverage map's
"strided = HARD / masked transpose" is **retired**: there is no masked-transpose work
to do. The remaining arbitrary-K items are completeness/perf, not correctness gates.

## See also
- [[arbitrary_k_tail_strategy]] · [[arbitrary_k_pad_vs_tail]] — the tail-handling set.
- [[arbitrary_k_codelet_coverage_map]] — the per-family map (strided row was "HARD"; this doc supersedes that for the *production* path).
