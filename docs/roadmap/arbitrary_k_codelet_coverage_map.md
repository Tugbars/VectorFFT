# Arbitrary-K Coverage Map — every codelet family, every feature

> Where the rem-aware tail (docs/performance/arbitrary_k_tail_handling.md) stands
> across the whole library, and what each remaining family needs. Source of truth:
> `generator/lib/coverage.ml` (the 6 quadrants × {avx2, avx512}) cross-referenced
> with the actual emitted signatures/loops.
>
> **The key structural fact:** every codelet batches over a *single* loop variable
> with simple strided access `arr[leg*stride + loopvar]` — `EXCEPT` the strided-2D
> family, which does a 4×4 AOS↔SOA transpose (loads VW whole rows). So the tail is
> the *same mechanism* (`current_ls_mode` + `emit_body` masked render) for every
> family but strided; strided needs a masked transpose.

## Status legend
- **DONE** — generated + validated bit-exact at odd K.
- **READY** — same strided pattern; the masking machinery (mode-aware `render_load`/
  `emit_store`) already covers it. Only the per-signature *structure* (hoist the loop
  var, emit the `if (var<bound)` tail block) needs wiring. Mechanical.
- **HARD** — transpose-based; the tail breaks the "VW full rows" assumption.

## The map

| Feature | Quadrant (files av2/av512) | Family / codelets | Signature & loop | Load/store pattern | Tail |
|---|---|---|---|---|---|
| **1D c2c in-place** | `inplace` (324/324) | n1, t1/t1s × dit/dif × fwd/bwd × flat/log3 (18-family) | `(rio_re,rio_im,tw,ios,me)` — `for k<me k+=VW` | `rio[leg*ios+k]`; spill scratch full-width | **DONE** (avx2; composite incl). avx512 emit-present, untested |
| **1D c2c OOP** | `oop` (52/52) | n1_oop, t1p_oop(+log3), **t1_oop (NEW)**, *_spec | `(in_re,in_im,out_re,out_im,tw,…,me)` — `for b<me b+=VW` | `in_re[b*gstride+leg*lstride]` → split out | ✅ **DONE 2026-06-28** |
| **Trig (DCT/DST/DHT)** | `trig` (36/36) | dct2/3/4, dst2/3/4, dht, dct1, dst1 | `(in,out,K)` 3-arg r2r — `for k<K k+=VW` | `in[leg*K+k]` → `out[…]` | ✅ **DONE 2026-06-28** (dev direct-codelet layer) |
| **1D r2c (real fwd)** | `rfft` (65/64) | r2cf leaf, hc2hc stage, hc2c / hc2c-nat terminator (+log3, +ranged) | `(in_re[,in_im],out…,is,os…,vl)` — `for v<vl v+=VW` | `in_re[leg*is+v]` → `out`/`Rp,Ip,Rm,Im` | **READY** (ranged + 4-buf split: see notes) |
| **1D c2r (real bwd)** | `c2r` (50/29) | r2cb leaf, hc2hc_dif_bwd stage, hc2c-nat initiator (+log3, +ranged) | `(in_re,in_im,out_re,is_re,is_im,os_re,vl)` — `for v<vl v+=VW` | `in_re[leg*is_re+v]` → `out_re[…]` | **READY** (ranged: see notes) |
| **2D c2c (row batch)** | `strided` (14/8) | n1 fwd/bwd strided | `(rio_re,rio_im,tw,row_stride,me)` — `for b<me b+=VW` | **4×4 transpose**: loads `rio[(b+0..b+3)*row_stride+col]` | **HARD** |

(2D and real-FFT also compose the 1D in-place codelets for their column/cascade
passes — those already have the tail. The rows above are the *additional* families
each feature introduces.)

## Why "READY" is mostly mechanical

The masking seam is `Isa.loadu_pd`/`storeu_pd` consulting `current_ls_mode`, reached
through `render_load` (the non-strided `else` branch — which OOP/trig/rfft/c2r all
take) and `emit_store`. Those are **already mode-aware**. What the in-place branch
has that the others don't is the *structure*:
1. hoist the loop var (`size_t v = 0; for (; v+VW<=vl; v+=VW)`),
2. after the loop, emit `if (v<vl){ rem=vl-v; <mask decl>; masked emit_body }`.

`emit_body` is already isa/mode-general; the `_vfft_masklo` table already emits for
avx2. So extending a family = generalize the bulk-loop-header + tail-block emission
(today hard-coded to `in_place`/`k`/`me`) to be parameterized by `(loopvar, bound)`,
then enable it per signature branch. One generalization, then flip it on per family.

### Per-family notes / wrinkles
- **OOP t1p / log3** twiddles are per-lane (`tw_re[…+b]`) → masked by the same thread;
  t1s-style broadcasts stay unmasked. `_spec` bakes strides as constants — unaffected.
- **rfft/c2r `vl`** is the batched lane count (independent real transforms / columns).
  rem==1 → scalar is fine (no spill in the leaves); the hc2hc stages are monolithic.
- **hc2c-nat** writes 4 split buffers (`Rp,Ip,Rm,Im`) — all indexed by `v`, all masked
  by `emit_store`; no extra work.
- **`ranged` variants** wrap the v-loop in an outer `hc_ranged` advance (pointer bumps
  after the loop, `emit_c.ml` ~3711). The tail block must sit *inside* the ranged loop,
  before the advance — a placement detail, not a new mechanism.
- **Real-FFT front doors** still fail-closed on `K%VW!=0 → NULL`; relax per feature once
  its codelets carry the tail.

### Why strided is HARD
The strided codelet loads VW *consecutive rows* (`rio[(b+0)*row_stride] … (b+3)*row_stride`)
and runs a 4×4 (avx2) / 8×8 (avx512) transpose to get lanes into SOA, then transposes
back to store. A remainder of `rem<VW` rows means the transpose has fewer than VW input
vectors. Options: (a) masked-load the `rem` rows (zero the rest) + masked-store back —
the transpose math is unchanged but the edge vectors are masked; (b) a scalar per-row
fallback for the `rem` rows. (a) is the vectorized path and matches the rest of the
design. This is the one family needing real new emit code.

## Suggested order (value × ease)
1. ~~**OOP c2c**~~ — ✅ **DONE 2026-06-28**. MODEB (scrambled) rides the in-place tail via the
   n1 OOP wrapper; LEAF (natural, N≤128) tailed n1_oop; BAILEY2 (natural, all N) got a NEW
   per-lane `t1_oop` codelet + per-group twiddle table (t1p's per-block broadcast straddles k2
   boundaries at odd K). codelet_oop.ml is a separate emit module — also fixed it hardcoding the
   obsolete M-fence ON. Validated vs naive DFT (natural order) + roundtrip.
2. ~~**Trig**~~ — ✅ **DONE 2026-06-28** at the *dev direct-codelet layer*. The dag r2r
   codelets (`radixN_{dct2,dct3,dct4,dst2,dst3,dst4,dht}_avx2`, N∈{8,16,32,64}) — the
   "coverage-complete" set in `trig_codelets.h`, registered via `trig_registry_avx2.h`,
   benched by `bench_trig_vs_fftw.c` — got the rem-aware tail (bulk `for(;k+VW<=K)` +
   scalar rem==1 with consts re-emitted inline as `double` + masked rem≥2). One wrinkle vs
   c2c: trig hoists function-scope `__m256d` trig constants, so the tail does
   `Hashtbl.reset hoisted_const_tags` to re-emit them at scalar/masked width. Validated by
   `build_tuned/test/test_trig_oddk.c`: 28 codelets × K∈{1,3,5,7} self-consistent vs even
   K=8 per-column (worst 8.9e-16, masked cols bit-exact) + even-K bulk vs direct FFTW
   formula (dct2/dst2/dct4/dst4/dht ~1e-13). **No front-door K guard on this layer** — the
   codelets are called directly `fn(in,out,K)`.
   ⚠️ **PRODUCTION trig is a SEPARATE path, NOT yet odd-K.** `vfft.h → vfft.c _build_trig`
   builds Makhoul **stride plans** (`stride_dct2_plan`/`stride_dct4_plan`/`stride_dht_plan`)
   over an inner `stride_r2c_plan(N,K,K, inner_c2c)` (DCT-IV: inner c2c only). build.py
   deliberately separates the two (the dag trig codelets are NOT in `dag_codelet_srcs()`).
   Production odd-K needs: (a) inner c2c odd-K [done], (b) `stride_r2c_plan` pre/post tail
   [= the **r2c** phase-2 item], (c) the trig-stride Makhoul twiddle/permute pre/post tail.
   **So production trig odd-K is gated behind r2c arbitrary-K.** DCT-IV (inner c2c, no r2c)
   is the one that could land first once its dct4-stride pre/post carries the tail.
3. **r2c / c2r** — ✅ codelets DONE 2026-06-28 (rfft cascade: r2cf/r2cb leaves, hc2hc stages,
   hc2c_nat terminator/initiator, +log3 +ranged — all carry the tail; in-place k-path proven
   byte-identical). Front door **selectively** relaxed (`src/core/transforms/real/`): rfft/natural
   guards opened to any K (`rfft.h`, `c2r.h`, `r2c_dispatch.h:245`, `c2r_dispatch.h:54`); the
   **decoupled-stride path kept K%8-gated** (`_vfft_r2c_build_stride` returns NULL at odd K — shared
   by r2c stride branch + c2r SPLIT; `c2r_dispatch.h:257` forces NATURAL at odd K). So odd K is
   routed onto the rfft cascade; the two bake-offs are already NULL-graceful → no bakeoff edit. The
   r2c stride is the SECOND strategy (vs rfft) — `n1_oop_strided` inner is vec-width-batched, can't
   do odd K yet. **These two stride gates are TEMPORARY** (see item 4).
4. **Strided 2D + decoupled-stride r2c/c2r** — HARD (masked transpose: the codelet loads VW whole
   rows + 4×4/8×8 transpose; a remainder of <VW rows needs masked-load the rem rows → transpose →
   masked-store). Unlocks odd row/col in 2D AND the decoupled-stride real-FFT path. **Doing this is
   what lets us DELETE the temporary stride gates from item 3** (`_vfft_r2c_build_stride` K%8 +
   `c2r_dispatch.h:257` force-NATURAL) — then the bake-off picks stride-vs-rfft at any K. Until then
   the gates correctly keep odd K off the transpose path.
5. **AVX-512** — emit path already present for all of the above; needs an AVX-512 host
   to test (this host is AVX2-only).
6. **Front-door guards** — relax `K%VW` → `K!=0` per feature as its codelets land.
