# R=32 AVX2: SPR vs Raptor Lake

Analysis after completing the R=32 bench on both chips. Comparing only AVX2
winners (Raptor Lake has no AVX-512) so the comparison is apples-to-apples.

---

## Headline numbers

| | SPR | Raptor Lake |
|---|---|---|
| Total decisions | 36 | 36 |
| Distinct winners shipped | 10 | 22 |
| buf wins | **9 (25%)** | **19 (53%)** |
| log3 wins | **14 (39%)** | **16 (44%)** |
| flat wins | 13 (36%) | **1 (3%)** |
| SW prefetch used | 28 (78%) | **33 (92%)** |
| drain `prefw` used | 8 (22%) | 6 (17%) |

## Three takeaways

### 1. Buffered explodes on Raptor Lake
From 25% → 53%. More than doubles.

### 2. Flat essentially disappears
From 36% → 3%. SPR loves `ct_t1_dit__tpf16r2` — it wins 11 of 36 decisions on SPR, almost always at me=256 and me=512 in the power-of-2-ios regime. On Raptor Lake it wins exactly 1 region.

### 3. SW prefetch becomes near-universal
92% vs 78%. The prediction that weaker HW prefetcher → more SW prefetch dominance held.

log3 holds steady at ~40% on both chips — it doesn't care which side of the server/consumer divide you're on.

---

## Tile size — the clearest µarch signal

**SPR (9 buffered wins):**

| Tile | Wins |
|---|---|
| tile256 | 7 |
| tile128 | 2 |
| tile32/64 | 0 |

SPR wants **big tiles**. Its wider front-end (6+ decoders, 8 execution ports) amortizes buffering overhead over more work per iteration.

**Raptor Lake (19 buffered wins):**

| Tile | Wins |
|---|---|
| tile256 | 8 |
| tile32 | 5 |
| tile128 | 3 |
| tile64 | 3 |

Raptor Lake uses **both extremes** with very different regime preferences:

| me | SPR buf tile | Raptor Lake buf tile |
|---|---|---|
| 64 | tile128, tile256 | tile128, tile256 |
| 128 | tile256 | tile256 (all 6 buf wins) |
| 256 | — (flat dominated) | tile64 (2) |
| 512 | — | tile32, tile64 |
| 1024 | — | **tile32** |
| 2048 | — | **tile32** |

At small me (64, 128), both chips prefer big tiles. The working set is small enough to live entirely in L1, and big tiles amortize fixed costs best.

At mid/large me (256+), the chips diverge sharply:
- **SPR** abandons buffering entirely and goes flat or log3
- **Raptor Lake** stays buffered but collapses to **tile32** for me ≥ 1024

This is a very specific pattern — Raptor Lake's narrower µop window benefits from tiny tiles when the working set gets large, while SPR either benefits or has no effect (hard to tell, since SPR just picks flat/log3 and ignores buf entirely in that regime).

---

## Split-point table (fwd direction, power-of-2 ios)

| me | SPR winner | Raptor Lake winner |
|---|---|---|
| 64 | `flat_dit__tpf16r2` | `buf_dit__tile128_temp__tpf32r1` |
| 128 | `buf_dit__tile256_temp__prefw__tpf16r2` | `buf_dit__tile256_temp__tpf16r2` |
| 256 | `flat_dit__tpf16r2` | `buf_dit__tile64_temp__prefw__tpf4r1` |
| 512 | `flat_dit__tpf16r2` | `buf_dit__tile64_temp__prefw__tpf4r1` |
| 1024 | `log3__tpf8r1` | `buf_dit__tile32_temp__tpf32r1` |
| 2048 | `log3` (no prefetch) | `buf_dit__tile32_temp__prefw__tpf4r1` |

**At power-of-2 strides** Raptor Lake goes fully buffered, SPR mostly doesn't. The stride-aliasing effect on Raptor Lake is bad enough that buffering (which isolates the load pattern via a local temp buffer) pays off handsomely. On SPR the aliasing penalty is smaller, so buffering's overhead isn't worth it — flat+prefetch wins instead.

---

## Split-point table (fwd direction, padded ios)

| me | SPR winner | Raptor Lake winner |
|---|---|---|
| 128+64 | `flat_dit__tpf16r2` | `buf_dit__tile256_temp__tpf8r1` |
| 256+64 | `flat_dit__tpf16r2` | `log3__tpf32r1` (fwd) / `log3__tpf4r1` (bwd) |
| 512+8 | `flat_dit` (no prefetch) | `log3__tpf32r1` |
| 512+64 | `log3` (no prefetch) | `log3__tpf16r2` |
| 1024+8 | `log3__tpf8r1` | `log3__tpf32r2` |
| 1024+64 | `log3` (no prefetch) | `log3__tpf32r1` |
| 2048+8 | `log3__tpf16r1` | `log3__tpf16r2` |
| 2048+64 | `log3__tpf16r1` | `log3__tpf16r2` |

At padded strides the two chips agree much more — both prefer log3 with some SW prefetch distance, converging on similar family choices. The exact prefetch distance differs (SPR likes tpf8, Raptor Lake likes tpf16/32) but the family is the same.

**The stride-aliasing penalty is what forces the divergence**, not some fundamental algorithmic preference.

---

## The stride-aliasing penalty in absolute terms

Same codelet, same me, different stride offsets — flat t1_dit fwd:

| me | stride pattern | SPR ns | Raptor Lake ns |
|---|---|---|---|
| 1024 | pow2 | high | high |
| 1024 | +8 padded | much lower | much lower |
| 2048 | pow2 | high | **very high** |
| 2048 | +8 padded | much lower | lower |

Both chips suffer at pow2 strides. Raptor Lake's absolute penalty at me=2048 is bigger — likely because its smaller L2 way-count efficiency and weaker prefetcher compound.

---

## drain_prefetch (`prefw`) pattern

SPR: 22% of decisions use `prefw`. Raptor Lake: 17%.

Counterintuitive at first — Raptor Lake has smaller DTLB coverage, so DTLB warming via prefetchw should matter more. But looking at where `prefw` wins on each chip:

**SPR prefw wins:** clustered at me=64, me=128 (working set barely fits in L1, output pages are new).

**Raptor Lake prefw wins:** me=64, me=128, and me=2048.

The me=2048 Raptor Lake wins (2 of them) suggest prefw matters for *large* working sets there too — consistent with Raptor Lake's DTLB pressure at large me. But for mid-me (256-1024), Raptor Lake picks non-prefw variants.

Hypothesis: Raptor Lake's HW prefetcher + small tile combination at mid-me already triggers DTLB fills via the access pattern itself, making explicit prefw redundant. Only when the tile gets small enough AND me gets large enough do both mechanisms matter.

---

## What this tells us about the project

Two chips, same codelet library, **completely different selectors**:

- SPR ships 10 codelets, dominated by `flat_dit__tpf16r2` (11 wins)
- Raptor Lake ships 22 codelets, dominated by various buffered tile sizes

A single "universal best codelet" library would ship either:
- A codelet optimized for SPR that costs Raptor Lake 1.5-2× in the buffered regions, or
- A codelet optimized for Raptor Lake that costs SPR 1.2× in the flat regions

Neither is acceptable when the point of having a codelet library is to be fast.

**The autotuning pays for itself.** The R=32 sweep takes ~100 seconds on Raptor Lake, ~6 minutes on a server under load. That's a one-time cost per (chip × library version). The resulting selector delivers optimal codelet choice for every (ios, me) point without runtime overhead.

Cross-chip comparison also confirms **the pruning rules are chip-agnostic** — stream drain correctly loses everywhere, cache-spill gating correctly excludes prefetch=0 only at tiny me. These can ship as hard rules in the generator.

What stays chip-specific: **tile size selection, prefetch distance selection, drain_prefetch enabling, family preference at each (me, stride) region**. That's exactly what the bench is for.

---

## For R=64 and beyond

The split between "obviously prune at generation time" vs "let the bench decide" is now data-grounded:

**Prune at generation time:**
- `drain=stream` when output_bytes < L2_size
- `twiddle_prefetch>0` when twiddle_bytes < L1D_size
- (optionally) `drain_prefetch=True` when me > some threshold of L3 pressure — but this one needs more data across chips before we commit

**Let bench decide:**
- Which family wins at each (ios, me)
- Which tile size — chip-dependent enough that any heuristic we pick will be wrong somewhere
- Which prefetch distance — varies per region within a single chip
- Power-of-2 vs padded stride differences — these are real and produce different winners

For R=64, the candidate matrix scales roughly linearly with radix × tile_count × prefetch_count. Probably 300-500 candidates pre-gating. The parallel-compile architecture that got R=32 down to 11 seconds on Raptor Lake scales — at 300 codelets that's probably 20-25 seconds on the same chip.

The only real question for R=64 is whether to start pruning based on what SPR and Raptor Lake show above (drop certain tile/prefetch combos that never win on either chip). My lean: not yet. Two chips isn't enough to establish "never wins" confidently. Run R=64 full-Cartesian on both chips first, then prune for R=128 based on three-chip data if the candidate count becomes painful.
