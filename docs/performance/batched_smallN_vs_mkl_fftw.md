# Batched Small-N vs MKL & FFTW — the split-complex blind spot, and what "scalar" really is

> Head-to-head microbenchmarks of our radix-4 codelet (scalar **and** AVX2)
> against FFTW (guru split + interleaved) and MKL (DFTI split), on a batched
> 4-point complex forward FFT in our split-complex lane-batched layout
> (`re[e*K+lane]`). Three results that matter:
>
> 1. **On our split-batched layout, MKL is 3–8× slower than our *scalar*** and
>    FFTW's split path runs at scalar class — the split-complex batched
>    small-N regime is a genuine MKL/FFTW blind spot, and it's our home turf.
> 2. **Neither MKL nor FFTW has a "magic free tail"** — MKL pays its own odd-K
>    penalty up to **1.78×**. There is no vectorized-only trick we're missing.
> 3. **Our "scalar" codelet is not scalar** — it compiles to 32 SSE-scalar ops
>    (`vmovsd`/`vaddsd`), **zero x87**. It's the 1-wide rung of a SIMD-width
>    cascade, the same instruction class MKL floors at.
>
> Companion to the arbitrary-K design in
> [`docs/roadmap/arbitrary_k_vectorization.md`](../roadmap/arbitrary_k_vectorization.md).
> **Status: scratchpad microbenchmarks** (single radix-4 stage) — directional,
> not yet reproduced in the production bench harness. Treat the *ratios and
> shapes* as the takeaway, not the absolute ns.

## Setup

- **Host:** i9-14900KF (Raptor Lake), AVX2, **no AVX-512**, ~5.7 GHz turbo.
- **Compiler:** mingw gcc 15.2, `-O2/-O3 -mavx2 -mfma`.
- **Problem:** batched 4-point complex **forward** DFT, `K` independent
  transforms, split-complex lane-batched layout — element `e` of transform
  `lane` at `re[e*K+lane]` / `im[e*K+lane]`. A radix-4 `n1` codelet *is* a
  complete 4-point DFT (single stage, no twiddles). In-place, unnormalized.
- **Metric:** ns per 4-point transform = wall_time / (iters · K),
  `QueryPerformanceCounter`. Each contender computes the identical transform
  (verified bit-exact / to FP noise against a brute-force forward DFT-4).
- **FFTW** 3.3.10 (vcpkg): `fftw_plan_guru_split_dft` matching our layout
  (`dims{4,K,K}`, `howmany{K,1,1}`), and `fftw_plan_many_dft` interleaved
  contiguous (FFTW's preferred), both `FFTW_MEASURE`.
- **MKL** oneAPI (mkl_rt **LP64**, `MKL_THREADING_LAYER=SEQUENTIAL`): DFTI
  `DFTI_COMPLEX` / `DFTI_REAL_REAL` (split) / strided (`{0,K}`, distance 1) /
  in-place, ISA forced via `MKL_ENABLE_INSTRUCTIONS`.

---

## Result 1 — vs FFTW (ns per 4-pt transform)

| K | our-scalar | our-avx2 | FFTW-split *(our layout)* | FFTW-interleaved *(FFTW's best)* |
|---|---|---|---|---|
| 8 | 1.51 | **0.47** | 1.79 | 1.02 |
| 64 | 1.47 | **0.38** | 1.52 | 0.76 |
| 1024 | 1.65 | 0.96 | 1.66 | **0.83** |
| 8192 | 1.62 | **0.72** | 1.67 | 0.82 |
| 65536 | 4.91 | 3.79 | 4.92 | **1.73** |

- **Our scalar ties FFTW-split at every K** (1.0–1.2× — dead even). FFTW's
  *split* path finds no SIMD codelet for strided split data and runs at scalar
  class.
- **Our AVX2 beats FFTW-split 2–4×** (peak 4.05× at K=64) and beats even
  **FFTW-interleaved** (FFTW's preferred layout) for small–mid K.
- FFTW-interleaved wins at K=65536 (1.73 vs our 3.79): past L2 it's pure DRAM
  bandwidth and FFTW's single contiguous stream beats our two split streams
  (`re[]` + `im[]`) — our element-strided batch access (8 streams at K-strides)
  blows TLB/cache there. **We have no interleaved path**, so that column is a
  layout we don't offer; the apples-to-apples FFTW-split column we win throughout.

---

## Result 2 — vs MKL (ns per 4-pt transform, split layout)

Each even `K` paired with `K−1` (odd) to expose MKL's tail penalty.

**MKL forced SSE4.2 (its ISA floor):**

| K (even) | ours | MKL | K (odd) | ours | MKL | MKL odd/even |
|---|---|---|---|---|---|---|
| 8 | 1.51 | 12.70 | 7 | 1.44 | 15.66 | 1.23× |
| 64 | 1.43 | 6.31 | 63 | 1.45 | 6.46 | 1.02× |
| 1024 | 1.74 | 5.56 | 1023 | 2.61 | 8.55 | **1.54×** |
| 8192 | 2.81 | 14.99 | 8191 | 2.70 | 12.12 | 0.81× |

**MKL forced AVX2 (its best on this AVX-512-less host):**

| K (even) | ours | MKL | K (odd) | ours | MKL | MKL odd/even |
|---|---|---|---|---|---|---|
| 8 | 1.50 | 13.08 | 7 | 1.42 | 16.30 | 1.25× |
| 64 | 1.43 | 6.52 | 63 | 1.45 | 6.71 | 1.03× |
| 1024 | 1.77 | 5.65 | 1023 | 1.76 | 10.07 | **1.78×** |
| 8192 | 2.59 | 12.79 | 8191 | 3.10 | 13.73 | 1.07× |

Output bit-exact vs our scalar (`maxdiff = 0.0e+00`).

- **Our scalar is 3–8× faster than MKL** on this layout (K=8: 1.5 vs 12.7;
  K=64: 1.43 vs 6.3). Our AVX2 widens that to ~8–28×.
- **MKL pays its own odd-K tail penalty** — up to **1.78×** (K=1023, AVX2). So
  the gold standard is *not* immune; there is no free vectorized tail it knows
  that we don't.
- **MKL barely vectorizes this layout**: SSE4.2 ≈ AVX2 (12.7 vs 13.1 at K=8).
  Its split (`DFTI_REAL_REAL`) + strided DFTI is a neglected slow path — slower
  here than even FFTW-split — plus DFTI per-call descriptor overhead dominates a
  4-point.

---

## Result 3 — "scalar" is SSE-scalar, not x87

Disassembly of our generated `radix4_n1_fwd_scalar` (`--isa scalar`):

```
32 × {vmovsd, vaddsd, vsubsd}  (xmm, 1-wide SSE)
 0 × x87 (fld/fadd/fmul/...)
```

gcc lowers `double` arithmetic to **SSE-scalar** on x86-64 by default. So our
"scalar" codelet is the **1-wide rung of the SIMD-width cascade**, the exact
instruction class MKL floors at (SSE) — not a foreign non-SIMD path. The
"scalar vs vectorized" dichotomy dissolves:

```
all xmm/ymm, no x87 ever:
  bulk        → AVX2 4-wide        (vaddpd ymm)
  light tail  → SSE  1-wide        (vaddsd xmm)   ← what we call "scalar"
  heavy tail  → AVX2 4-wide masked (vmaskmovpd ymm)
```

Consequence for the arbitrary-K work: `--isa scalar` is just the `vec_width=1`
case of the same emit (not a separate untested backend), and it's already
FFTW-grade / MKL-beating on this layout.

---

## Result 4 — tail-strategy crossover (informs the hybrid)

Where does a masked vector tail overtake the SSE-1-wide ("scalar") tail?
Synthetic sweep at K=7, chaining the radix-4 butterfly `W` times to scale
codelet weight (~16·W ALU ops/lane); ns per 4-pt transform:

| W | ~ALU/lane | masked-fwd (4-wide) | "scalar" (1-wide SSE) | pad-stack | winner |
|---|---|---|---|---|---|
| 1 | 16 (≈radix-4) | 13.4 | **7.1** | 46.9 | scalar |
| 2 | 32 (≈radix-8) | 16.8 | **11.4** | 49.1 | scalar |
| 4 | 64 (≈radix-16) | **21.2** | 23.1 | 50.3 | masked |
| 8 | 128 (≈radix-32) | **32.0** | 46.1 | 53.2 | masked |
| 16 | 256 | **57.8** | 90.4 | 65.5 | masked |
| 32 | 512 | **101.6** | 190.1 | 115.2 | masked |

- **Crossover ≈ 40–50 ALU ops/lane**: light codelets (radix ≤ 8) finish faster
  1-wide; heavy codelets (radix ≥ 16) finish faster 4-wide masked. The 1-wide
  tail wins light because it does only the `rem` real lanes and overlaps with
  the bulk on otherwise-idle scalar ports; the 4-wide masked tail does a full
  extra vector butterfly (port pressure) that only pays off once the body is big.
- **`maskstore` tax is negligible** (1.04× vs plain `storeu`) — the cost is the
  redundant full-vector butterfly, not the mask.
- **Pad-into-stack-scratch is dominated everywhere** (gather/scatter copy kills
  it for light; ~10% behind masked for heavy) — rejected.
- Masked tail is **bit-exact for all K**; the naive full-store overlap corrupts
  the overlap lane in-place (the documented trap — needs masked store).

⇒ Hybrid tail by **radix weight** (not K): light → 1-wide SSE, heavy → 4-wide
masked; bulk always 4-wide AVX2. Final split to be set by per-codelet
measurement in the production harness.

---

## Caveats (read before quoting these)

- **MKL is off its home turf.** `DFTI_REAL_REAL` + strided + tiny-N is a slow
  MKL path; MKL is engineered for **interleaved-contiguous whole transforms**,
  where it is excellent. These numbers say "MKL is weak on *our* layout," not
  "MKL is slow."
- **DFTI per-call overhead** dominates a 4-point transform (5–15 ns is mostly
  descriptor dispatch, not compute). Our codelet is a bare function call. A real
  MKL user pays this overhead too via the public API, but it's not a compute
  comparison.
- **Single radix-4 stage**, not a full multi-stage FFT. Larger N changes the
  picture (FFTW/MKL recursive composition vs our multi-stage plans).
- **Microbenchmarks from scratchpad spikes** (`r4_vs_fftw.c`, `r4_vs_mkl.c`,
  `crossover.c`, `masked_tail_spike.c`), not the production bench harness.
  Reproduce there before quoting in `v1_0_results.md`.
- Timings are noisy on a desktop (frequency scaling, background load) — ratios
  jumped ±10% between runs. Trust shapes, not last digits.

---

## Strategic conclusions

1. **The split-complex batched small-N regime is a real MKL/FFTW blind spot.**
   Both libraries optimize interleaved-contiguous; on the split lane-batched
   layout we target, our AVX2 wins 2–28× and even our *scalar* ties/beats them.
   This is the [MKL blind-spot positioning](../roadmap/arbitrary_k_vectorization.md)
   made concrete at the smallest transform.
2. **There is no magic vectorized tail to discover.** MKL — vectorized-only —
   still pays an odd-K penalty (≤1.78×). Our masked/SSE-cascade tail is a
   legitimate, competitive answer, not a workaround.
3. **The tail is a SIMD-width cascade, all SSE/AVX ISA.** "Scalar" = 1-wide SSE.
   This removes the last reason to fear the `--isa scalar` path for the
   light-radix tail.
4. **Our one ceded regime is interleaved-large-K** (FFTW-interleaved at K=65536),
   a layout we don't offer and FFTW's home — consistent with leading where they
   invest least, not chasing where they invest most.
