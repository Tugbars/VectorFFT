# Arbitrary-K tail strategy — output order decides the remainder mechanism

> **Thesis (Tugbars, 2026-06-29):** SSE2/scalar tail handling is the remainder
> mechanism **of the scrambled / transposeless path only**. Natural-order plans have
> a reorder stage, and the remainder belongs to *that* stage (the strided / masked
> transpose), not to a per-lane SSE2 tail. Where the tail goes is decided in the
> **plan phase**, by the output-order the user selected.

This is the organizing principle behind which families get the per-lane SSE2/scalar
tail and which should (eventually) get a transpose-based remainder. It also explains
why benching the SSE2 tail on the natural rfft path gave garbage — wrong vehicle on
every axis.

## The two paths

Our batched FFT packs the **K batch across the SIMD lanes** in a **split (SOA) re/im
layout**. Whether a transpose/reorder ever happens is what splits the world:

### 1. Scrambled / transposeless  →  SSE2 → scalar tail (the real mechanism)

The in-place c2c engine is **transposeless by design**
([engine/README.md:56-70](../../../src/core/engine/README.md#L56-L70)): "no scratch, no
ping-pong, no bit-reversal, and no transpose… the K batch is contiguous in memory so
no transpose is needed to feed the vector units." The cost is the native output order
([README.md:86](../../../src/core/engine/README.md#L86)): **forward output is
digit-reversed (scrambled)**. So *scrambled is what the transposeless path naturally
produces*; natural order is the thing that **costs** a reorder.

Members: **in-place c2c**, and OOP **MODEB** ("in-place stride engine, OOP-adapted,
scrambled digit-perm" — [oop/README.md:30](../../../src/core/oop/README.md#L30)).

Because the batch K stays **SOA / lane-packed end-to-end**, there is no point at which
the leftover `K mod VW` transforms become contiguous-per-transform. The remainder
lanes are *stuck* in the packed layout, so the **only** way to finish them is in that
layout at narrower width:

```
bulk:  for (; b + VW <= K; b += VW)   full-width (AVX2 = 4 lanes)
tail:  if (b < K) { rem = K - b;
         rem == 1 → scalar single lane
         rem >= 2 → SSE2 width-2 loop + scalar straggler }   // AVX2; AVX-512 = masked
```

This is the tail's **true home**: it is the *only* correct remainder method here, it
is where we **beat MKL (1.5–2.1×)** — the MKL blind spot ([[mkl_blind_spot_positioning]])
— and odd K is **calibratable** on this path (`plan_create_ex` has no K%VW guard).
Measure the SSE2 tail's performance here (`bench_oddk_tail`), nowhere else.

### 2. Natural / transposed  →  remainder belongs to the transpose

Natural-order plans contain a reorder stage: OOP **LEAF** (single column-layout
codelet) and **BAILEY2** ("transpose fused into the stores",
[oop/README.md:28-29](../../../src/core/oop/README.md#L28-L29)), and the real-FFT
families (rfft / c2r / trig Makhoul), which reorganize between split-batched and
per-transform layouts.

A reorder is **exactly** what lets you **deinterleave the `K mod VW` remainder into
contiguous-per-transform form and run it per-K** — which is MKL's whole trick: with an
interleaved, per-transform-contiguous layout there is *no SIMD-batch remainder at all*
(odd K is "just one more transform"). So on a transposed path the remainder is properly
owned by the **strided / masked transpose** (the phase-2 endgame), not by a per-lane
SSE2 tail.

On these families today the SSE2 tail is a **correctness stopgap** — bit-exact for all
K (validated), shipped so odd K works *now* — but it is **not the perf design**. When
the strided/masked transpose lands it **supersedes** the SSE2 tail on the natural
families (this is the same work that "removes the temporary r2c/c2r stride gates" in
[[arbitrary_k_codelet_coverage_map]]).

## Why (the deeper reason)

The split + lane-packed layout is a **deliberate** trade: it costs a split-layout tax
single-threaded (MKL's interleaved per-transform layout has no batch remainder and
stays L1-resident) but **pays off multithreaded** (no per-transform setup, perfect
streaming, K-split scales). See [[r2c_st_loses_mt_wins]] and [[memory_bound_thesis]].
The scrambled path commits fully to SOA-throughout (max MT benefit, no reorder) and so
*must* solve the remainder in-layout → SSE2/scalar. The natural path already pays for a
reorder, so it can fold the remainder into that reorder for free.

## Consequences (the operational rules)

1. **SSE2/scalar tail ⇒ scrambled path only** (in-place c2c, MODEB). That is its home,
   its only-correct-mechanism status, its MKL win, and its calibratable odd K.
2. **Bench the SSE2 tail on scrambled cells** via `bench_oddk_tail` with calibrated
   odd-K wisdom — **never** on natural rfft ST. The rfft path is structurally
   ST-losing *and* its odd K is uncalibratable (`calibrate_r2c.c` rejects `K%8 != 0`),
   so any odd-K rfft-vs-MKL number is an uncalibrated default-plan artifact (the
   N=512 K=16 @0.37× garbage).
3. **Natural families keep the SSE2 tail as a stopgap** until the strided/masked
   transpose lands; then the transpose owns the remainder and the SSE2 tail retires on
   those families.

## See also
- [[arbitrary_k_vectorization]] — the productionized tail (§ 2026-06-29 SSE2 pivot).
- [[arbitrary_k_scalartail_experiment]] — full experiment record + the SSE2-vs-masked
  bake-off and robust-bench methodology.
- [[arbitrary_k_codelet_coverage_map]] — which families carry the tail; strided endgame.
- [engine/README.md](../../../src/core/engine/README.md) — transposeless / scrambled by design.
- [oop/README.md](../../../src/core/oop/README.md) — LEAF/BAILEY2 (natural) vs MODEB (scrambled).
