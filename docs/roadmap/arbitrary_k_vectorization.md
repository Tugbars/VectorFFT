# Arbitrary-K Vectorization — Two Strategies and the Split-Layout Through-Line

> Why the library currently only accepts batch widths that are a multiple of the
> SIMD width, what breaks at `K=7` and `K=1`, and the two complementary
> strategies that close the gap — a **masked tail** for odd `K ≥ VW` and a
> **within-transform** path for `K < VW`. The unifying observation: our
> split-complex layout is an advantage in *both* strategies, for the same
> structural reason.
>
> Companion to the executor in
> [`engine/stride_executor.h`](../../src/core/engine/stride_executor.h) and the
> codelet emitter in
> [`generator/lib/emit_c.ml`](../../src/dag-fft-compiler/generator/lib/emit_c.ml).

## Frame

The library vectorizes **across the batch (K) dimension**: each SIMD lane holds
one independent transform, data laid out `data[e*K + lane]`. The generated
butterfly loop ([`emit_c.ml:1612`](../../src/dag-fft-compiler/generator/lib/emit_c.ml#L1612))
is:

```c
for (size_t b = 0; b < me; b += VW)   /* VW = 4 (AVX2) | 8 (AVX512); NO remainder loop */
```

`VW` is `Isa.vec_width` ([`isa.ml`](../../src/dag-fft-compiler/generator/lib/isa.ml):
`4` for AVX2, `8` for AVX-512, `1` for the scalar fallback). The loop steps by
`VW` and **has no tail**, so the per-codelet lane count `me` must be a multiple
of `VW` or the final iteration overruns into the next element's storage —
**silent corruption, not a crash.**

Two distinct gaps fall out of this:

| Gap | Example | Today's behavior |
|---|---|---|
| **Odd `K ≥ VW`** — a remainder in the batch | `K=7`, `K=19` | corruption on in-place c2c; the real-FFT/OOP dispatchers fail-closed (`K % 8 != 0 → NULL`) |
| **Tiny `K < VW`** — batch smaller than one vector | `K=1`, `K=3` | can't fill even one vector; same overrun / rejection |

The defensive `K % 8 != 0 → NULL` guards live in
[`oop_plan.h`](../../src/core/oop/oop_plan.h),
[`r2c_dispatch.h`](../../src/core/transforms/real/r2c_dispatch.h),
[`c2r_dispatch.h`](../../src/core/transforms/real/c2r.h), and
[`rfft.h`](../../src/core/transforms/real/rfft.h). They are a *guard*, not an
answer — the user who pushes `K=7` gets nothing.

> Note on `% 8` vs `% 4`: the AVX2 codelet only *needs* `K % 4 == 0`. The stricter
> `% 8` guard is a **wisdom-portability** policy (a plan tuned at `K%8==0` is valid
> on AVX-512 too), not a codelet requirement.

---

## The two vectorization axes

There are exactly two places to find SIMD parallelism in an FFT. We use one;
FFTW/MKL use both. The contrast is the whole story, so here it is concretely on
a radix-2 butterfly (`out0 = a + b`, `out1 = a − b`).

### Axis 1 — across-batch (what we ship): a lane = a whole transform

`K` independent FFTs. One register packs *the same element from VW different
transforms*:

```
reg_a = [ x0·FFT0 | x0·FFT1 | x0·FFT2 | x0·FFT3 ]
reg_b = [ x4·FFT0 | x4·FFT1 | x4·FFT2 | x4·FFT3 ]
add  → [ (x0+x4)·FFT0 | …FFT1 | …FFT2 | …FFT3 ]    ← 4 transforms, one instruction
```

Each lane is a separate problem; the transforms never interact. The arithmetic
is identical to scalar code, just widened — **no shuffles, ever.** The price:
you need `K ≥ VW` transforms to fill the register. At `K=1`, three lanes are air.

### Axis 2 — within-transform (FFTW/MKL; proposed for `K=1`): a lane = a point

One FFT. The register packs *different points of the same transform*:

```
reg_a = [ x0 | x1 | x2 | x3 ]    ← points of ONE transform
reg_b = [ x4 | x5 | x6 | x7 ]
add  → [ x0+x4 | x1+x5 | x2+x6 | x3+x7 ]    ← 4 butterflies of one transform, one instruction
```

`K=1` and the register is **full** — the parallelism comes from inside the
transform. But the access stride changes across stages: stage 0 pairs points 4
apart (separate registers, clean); a later stage pairs points 2 apart, which now
sit *in the same register* → you must **shuffle/permute** to line the partners up
before adding. Every stage whose butterfly stride drops below `VW` needs
permutes. That's the cost FFTW's genfft exists to manage.

| | lane = | shuffles | fills at K=1? | wins at |
|---|---|---|---|---|
| **across-batch** | a whole transform | **none** | no (lanes idle) | large batch, memory-bound throughput |
| **within-transform** | a point of one transform | every sub-VW stage (+ cmul, in interleaved layouts) | yes | single-transform latency |

---

## The through-line: split layout decouples re/im from the lane axis

This is the architectural observation that makes *both* strategies pay off, and
it is the reason to write any of this down.

Complex multiply is the operation that forces shuffles —
`(a_re·w_re − a_im·w_im) + i(a_re·w_im + a_im·w_re)` requires all the `re`s
together and all the `im`s together. In an **interleaved** layout
(`[re,im,re,im]`, FFTW's ABI) `re` and `im` live *inside* the vector, so a cmul
needs a shuffle **no matter what the lanes represent**. We keep `re[]` and `im[]`
in **separate arrays**, so the cmul is just FMA pairs and the lane axis is free
to carry whatever we want:

| | lane axis carries | cmul | extra benefit |
|---|---|---|---|
| **across-batch** | different transforms | shuffle-free | packs full `VW` transforms/reg (interleaved packs `VW/2`) |
| **within-transform** | points of one transform | shuffle-free | **only stage-transition shuffles remain** — FFTW pays those *plus* cmul shuffles |

> **Split layout puts re/im on a separate axis from the SIMD lanes; interleaved
> layout puts them on the lane axis.** Because our advantage lives on the *other*
> axis from the lanes, it is orthogonal to the strategy: across-batch,
> within-transform, `K=1`, `K=256` — the complex arithmetic stays clean in all of
> them. FFTW cannot pick this up without abandoning interleave, which is baked
> into its codelet ABI. One layout decision compounds in every vectorization mode
> we will ever add.

---

## Strategy 1 — masked tail for odd `K ≥ VW` (across-batch)

For `K` with a remainder (`K=7`, `K=19`), keep the across-batch engine and add a
single tail pass so the codelet handles any `me`.

### Shape: bulk loop + one masked tail pass

```
bulk: for (b = 0; b + VW <= me; b += VW)  { … full vectors … }   ← unchanged hot path
tail: if (b < me)  { one more full-width vector, MASKED store }   ← new
```

For `K=7` (AVX2): bulk does lanes 0–3, tail does lanes 4–6. Two passes total,
both full-width SIMD, **no scalar**. For `K=19`: bulk does lanes 0–15, tail does
16–18 — "bulk + tail" is always two logical phases regardless of `K`.

### The in-place trap (why a *full-store* overlap is wrong)

The naive tail is an **overlap**: step the base back to `me - VW` so the load is
full-width and in-bounds (for `K=7`, read lanes 3–6, lane 3 redundant). For an
out-of-place single-write kernel that's fine — lane 3 is written twice with the
same value. **It corrupts the in-place path.** Concretely, radix-2, one stage,
`K=7`:

1. Bulk transforms lanes 0–3 *in place* — `buffer[·*K + 3]` now holds butterfly
   output.
2. Overlap (base 3) re-reads lane 3. Its input is **already transformed**, so a
   full store writes a doubly-transformed value back → lane 3 corrupted. Lanes
   4–6 read original input → fine (lanes are independent).

The fix is a **masked store**: compute the full tail vector, store only the
genuinely-new lanes (mask out the overlap lane). The masked-out lane's stale
input is read and computed but discarded; lane independence means it can't poison
the valid lanes.

> **Rule: bulk loop + masked-store tail, never a plain full-store overlap.** On
> AVX-512 the mask is a convenience; on our in-place AVX2 path it is a
> *correctness* requirement.

### In-place vs OOP — the same masked tail serves both

The OOP executor is **not** pure ping-pong. MODEB
([`oop_execute.h:59-74`](../../src/core/oop/oop_execute.h#L59-L74)) runs **stage 0
out-of-place** (`src→dst`), then **stages 1.. in-place on `dst`**. So OOP is
mostly in-place and cannot dodge the trap. One masked-tail codelet is correct for
both placements:

| | overlap lane read | masked store does | verdict |
|---|---|---|---|
| **in-place** (`in==out`) | already-transformed (bulk wrote it) | masks it out → not re-stored | correct — **mask required** |
| **OOP** (`in≠out`) | original input (`in` never written) | masks it out → bulk wrote it once | correct — clean single write |

In both, the new lanes `[floor·VW, me)` are read from untouched input and stored
correctly. We **do not fork the codelet by placement** — only by ISA.

### ISA: forward-masked (AVX-512) vs overlap-back maskstore (AVX2)

| | tail mechanism | mask | overlap? |
|---|---|---|---|
| **AVX-512** | `_mm512_maskz_loadu_pd` + `_mm512_mask_storeu_pd` | `__mmask8 = (1u<<rem)-1` | no — masked load reads only valid lanes (no fault), go forward |
| **AVX2** | plain `loadu` + `_mm256_maskstore_pd` | `__m256i` sign-bit vector for the top `rem` lanes | yes — base `me - VW` keeps the load in-bounds; maskstore is the **only** masked op |

AVX2 also has `_mm256_maskload_pd`, so the forward form works there too; we prefer
**overlap-back** because it needs maskstore only (cheap plain load, sidesteps any
maskload quirks). On Raptor Lake `maskstore` is ~1–2 µops — not the Haswell-era
penalty.

### Wrinkle: log3 (t1p) twiddles assume `me % VW == 0`

The per-position broadcast twiddle table is `(R-1)*(me/VW)` blocks, addressed
`tw_re[j*(me/VW) + b/VW]`
([`emit_c.ml:169`](../../src/dag-fft-compiler/generator/lib/emit_c.ml#L169)). At
`me=7`, `me/VW = 1` — lanes 4–6 have no twiddle block. For odd `K` either pad that
table to `ceil(me/VW)` blocks (planner change), or **don't select log3 on the
tail-bearing stage** (simplest for v1). `n1`/`t1`/`t1s` use per-lane or per-leg
twiddles and are unaffected.

### Scope

- [`isa.ml`](../../src/dag-fft-compiler/generator/lib/isa.ml): add a
  `maskstore_pd` intrinsic + a mask-builder per ISA.
- [`emit_c.ml:1612`](../../src/dag-fft-compiler/generator/lib/emit_c.ml#L1612):
  split the loop into full-vector bulk + a masked overlap-back tail that reuses
  the existing body at base `me - VW`.
- Relax the `K % 8` dispatch guards to `K != 0` once codelets handle any `me`.
- MT slicer ([`threads.h`](../../src/core/support/threads.h) + the per-feature
  K-split wrappers): the slice is currently rounded to a multiple of 8 (one cache
  line, no false sharing); let the **last** worker carry the non-aligned
  remainder and the codelet's masked tail absorbs it.
- Prove spike-first on one codelet (radix-4 `n1` AVX2), `K=7` in-place
  forward+backward roundtrip to 1e-14, vs a scalar-tail baseline, *then* roll
  across the generator — mirrors the natural-c2r arc.

---

## Strategy 2 — within-transform for `K < VW` (especially `K=1`)

For batches smaller than one vector, across-batch can't fill the lanes. Add the
second axis: vectorize across one transform's own points. This is FFTW's home
turf — and **our split layout makes it cheaper than FFTW's.**

### Why split makes it cheaper than FFTW

FFTW pays shuffles for two reasons: stage-transition butterflies *and* complex
multiplies (interleave forces a re/im separation per cmul). In split layout the
**cmul shuffles vanish** (see the through-line above) — a 4-point complex mul is:

```
out_re = a_re*w_re − a_im*w_im     ← FMA pair, lanes = 4 points, no shuffle
out_im = a_re*w_im + a_im*w_re
```

The **only** shuffles left are stage-transition, and those bound to the last
`log_radix(VW)` levels. A **radix-VW leaf** (radix-4 on AVX2, radix-8 on
AVX-512) does those in-register butterflies once with the permutes baked into
that one codelet; everything coarser than `VW` composes shuffle-free. So our
within-transform leaf is leaner than FFTW's by exactly the cmul-shuffle count.

### A new lowering backend, not a new engine

The DAG and scheduler are unchanged. What changes is the **lane mapping**: today
a lane is a transform (`arr[j*ios + b]`, loop `b += VW`); in within-transform a
lane is a *point*, and the leaf emits the in-register permutes. Plus a
within-transform executor (leaf + shuffle-free twiddle stages) and a dispatch
rule routing `K < VW` here. The generator already has a `scalar` ISA
([`isa.ml:71`](../../src/dag-fft-compiler/generator/lib/isa.ml#L71)) — that is the
*correct-but-slow* `K=1` path; within-transform is the same DAG lowered to real
SIMD instead.

### Staged

1. **Leaf-only, small N + `K=1`** — a single radix-VW leaf covers small single
   transforms (the common latency case) for modest effort and proves the
   lowering.
2. **Multi-stage within-transform** (shuffle-free composition for large `N` at
   `K=1`) — only if the large-`N`/`K=1` case proves to matter.

---

## How FFTW / MKL handle this (context)

> **Measured** (2026-06-24):
> [`docs/performance/batched_smallN_vs_mkl_fftw.md`](../performance/batched_smallN_vs_mkl_fftw.md)
> — on our split-batched 4-pt layout our *scalar* beats MKL 3–8× and ties
> FFTW-split; MKL pays its own odd-K penalty (≤1.78×), so there is no magic
> vectorized tail. And our "scalar" codelet is SSE-scalar (32 `vmovsd`, 0 x87) —
> the 1-wide rung of a SIMD-width cascade, not a non-SIMD path.

They do **not** scalar-fall-back the whole transform. They have **both axes** and
a measuring planner that composes mixed solvers, so they never reject and never
go all-scalar:

- **Across-batch preferred** when a batch exists (cleaner, no shuffles — same as
  us). So FFTW *does* hit the same "vector loop not a multiple of VL" situation.
- **Scalar codelets for the cleanup tail only** — genfft emits them; the planner
  uses one for the few remainder lanes when that measures cheapest.
- **Buffered solver** — copy an awkward chunk into aligned, padded scratch, run
  the fast codelet, copy back (the "pad-into-a-buffer" option, used for bad
  alignment *and* remainders).
- **Within-transform fallback** — keeps even `howmany=1` vectorized.

Our masked tail is **better than their remainder paths for the batched case**: no
copy, no scalar lane — the remainder runs at full SIMD width. It is the technique
hand-tuned batched kernels and GPU FFTs (cuFFT/VkFFT) use: predication/masking on
the tail. The one thing they have that we don't is the within-transform axis for
`K=1` — which Strategy 2 adds.

---

## Complete-K coverage

With both strategies there is no `K` the library handles poorly:

| K | path | lane occupancy |
|---|---|---|
| `≥ VW`, any (incl. odd) | across-batch + masked tail | full |
| `< VW` (2, 3) | within-transform, looped per transform | full |
| `1` | within-transform leaf | full — and we already beat them on overhead, so this wins decisively |

The dispatch boundary is the `K ≈ VW` crossover (open question below).

---

## The tiny-K floor — honest accounting

Two costs hide here and must not be conflated.

1. **The masked tail's own overhead** = the few redundant lanes in the *one* tail
   vector — at most `VW-1` lane-slots *total*, independent of `K`. For `K=255`
   that's 1 wasted lane in 256 — noise; amortizes to nothing as `K` grows. Not a
   concern. *But:* at low `K` the tail is a large fraction of the work (`K=7` → one
   of two passes), so the tail must stay genuinely lean (maskstore, no scalar, no
   branch in the bulk path) — it matters most exactly where we win.

2. **The tiny-`K` floor** is unrelated to the tail. At `K=1` the *entire workload*
   is smaller than one register; no tail trick can help, because there aren't
   `VW` transforms to pack. Across-batch runs at `K/VW` of our own ceiling
   (AVX2: `K=1`→25%, `K=2`→50%, `K=3`→75%). This is what Strategy 2 removes.

**Crucially, low occupancy ≠ losing.** Occupancy is efficiency relative to *our
own* peak; the competitive outcome is absolute time vs FFTW/MKL. At low `N` the
fight is won on **per-call overhead**, where we are lean (precompiled codelet
dispatch, split layout, no descriptor / plan-tree traversal) and they are not.
25% of a 5 ns kernel still beats 100% of a 40 ns dispatch — which is why the
measured results already win at low `K` / low `N`.

**MT amplifies this at low `K`.** Their threading is the weak point at small `N`
(FFTW vcpkg is ST-only; MKL doesn't parallelize small batched transforms — the
documented MKL-T8-slower-than-T1 anomaly), while our spin-pool wakes in ~10 ns and
K-splits with near-zero setup. Our scaling curve climbs while theirs is flat or
negative — "we beat even harder."

---

## Roadmap (staged)

1. **Masked tail** (Strategy 1). Closes *correctness* for every `K` immediately —
   including `K=1`, just at low occupancy. Banked fast; spike-first on radix-4.
2. **Within-transform leaf** (Strategy 2, small-N + `K=1`). Turns `K=1` from
   "correct" into "wins" — full SIMD *and* lower dispatch overhead than FFTW.
3. **Multi-stage within-transform** — large-`N`/`K=1`, only if it proves to matter.

## Open design decisions

- **Crossover `K`.** Where does dispatch switch from across-batch+masked-tail to
  within-transform? Likely `K ≈ VW`, but `K=2,3` could go either way (partial
  across-batch pass vs per-transform within-transform loop) — settle by
  measurement.
- **Shuffle strategy for the leaf.** Radix-VW leaf with baked-in permutes
  (preferred, bounded) vs explicit per-stage permutes.
- **log3 twiddle table** for odd `K` — pad to `ceil(me/VW)` blocks, or exclude
  log3 from the tail-bearing stage (v1).
- **Wisdom / JIT / MT wiring** for the within-transform path — mirror how the
  across-batch engine threads through `vfft.c` (calibrate → wisdom → dispatch →
  execute), so the new axis is a first-class citizen, not a bolt-on.
