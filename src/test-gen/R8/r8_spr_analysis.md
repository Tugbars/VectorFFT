# R=8 codelet bench results on SPR

Six-candidate matrix (ct_t1_dit, ct_t1_dif, ct_t1_dit_prefetch) × (avx2, avx512)
run on the Anthropic SPR container. VTune predictions on Raptor Lake
(memory-tier optimizations regress, DIF ~10% faster than DIT) replicate
on SPR — more strongly in the case of DIF dominance.

---

## Headline

72 decisions total across both ISAs, 36 per ISA.

| Family | SPR AVX2 | SPR AVX-512 |
|---|---|---|
| `ct_t1_dif` | **34** (94%) | **25** (69%) |
| `ct_t1_dit` | 2 (6%) | 11 (31%) |
| `ct_t1_dit_prefetch` | **0** | **0** |

### DIF dominates even more than VTune predicted

VTune measured DIF 10% faster than DIT on Raptor Lake at R=8 K=256.
On SPR the DIF win is similar or larger across all me values. At
AVX2 me=256 padded ios, DIF is 693 ns vs DIT's 852 ns — **23% faster**.
The gap narrows on AVX-512 but DIF still wins 69% of regions.

### Prefetch loses everywhere

VTune measured aggressive SW prefetch at +15% regression and lightweight
SW prefetch at +8% regression on Raptor Lake R=8. On SPR the prefetch
variant loses 100% of regions. Across all 72 sweep points, there is not
a single (ios, me, dir) combination where prefetch wins.

This is the strongest possible "negative control" result: the VTune
conclusion generalizes cleanly to a different µarch. R=8 is genuinely
compute-bound (dependency-chain-bound, specifically) and SW prefetch
adds pure front-end overhead with zero latency to hide.

**Specific regression magnitudes (SPR AVX2, fwd, padded ios):**

| me | dit | prefetch | regression |
|---|---|---|---|
| 64 | 239 ns | 271 ns | +13% |
| 128 | 468 ns | 574 ns | +23% |
| 256 | 852 ns | 1007 ns | +18% |
| 512 | 2162 ns | 2379 ns | +10% |
| 1024 | 4075 ns | 5022 ns | +23% |
| 2048 | 9533 ns | 10921 ns | +15% |

Consistent 10-23% regression across the full me range on a different
chip from where the VTune analysis was done.

## Per-me regime

**SPR AVX2:** DIF dominates at every me. At me ∈ {1024, 2048} DIT wins
one region each (both at pow2 stride where the DIT/DIF gap narrows).

**SPR AVX-512:** More mixed. DIT wins roughly 1/3 of regions, mostly at
me=512. AVX-512's 8-wide vector processes R=8 in one iteration per butterfly
(where AVX2 takes 2), so the critical path structure differs enough that
DIT occasionally wins.

Across both ISAs combined (72 decisions): DIF 82%, DIT 18%, prefetch 0%.

## What this tells us about the coverage model

### The "R=8 is compute-bound" hypothesis is confirmed

VTune showed R=8 has:
- Memory Bound: 1.4% (vs 30% at R=16, 66% at R=32)
- Core Bound: 23.4%
- L1 Latency Dependency: 36.8% of clockticks (dependency chains)

The bench now confirms this from the other direction: every memory-
targeted optimization is either neutral (DIT) or actively harmful
(prefetch). The only codelet-level optimization that helps is
**algorithmic** — DIF has a shorter critical path than DIT, which matters
because the bottleneck is exactly critical-path length.

### t1s would probably not help on R=8

We decided not to port t1s to R=8 because VTune said loads weren't the
bottleneck. This bench is consistent with that call:

- **If loads were the bottleneck**, DIT+prefetch would at least tie DIT.
  Instead it regresses 10-23%. No hidden load latency to hide.
- **Load count at R=8 is small**: 7 twiddle rows × me columns. At me=64,
  twiddle bytes = 7×64×16 = 7 KB, fits in L1 easily. At me=2048, 2.2 MB
  — bigger but the ROB sees far enough ahead to prefetch naturally.

Porting t1s here would add engineering surface without measurable win.
The coverage-doc claim "no compute-bound coverage for R=8" is fine —
the compute-bound codelet pattern IS DIF (vs DIT), and we cover both.

### The R=8 bench matrix should stay small

Six candidates, no knobs. The VTune-guided decision to prune tile/buf/
prefetch variants entirely was correct. Adding them would have tripled
the matrix size with every candidate losing to DIF.

## Comparison across all three radixes on SPR

| R | Winner family (SPR AVX-512) | Bottleneck |
|---|---|---|
| 8 | DIF (69%) | Critical-path length |
| 32 | t1s (50%) | Load-port pressure |
| 32 | flat (33%) | None (headroom) |
| 32 | log3 (17%) | Load-port pressure (partial) |

Three distinct bottlenecks show up depending on radix:
- **Small R**: compute-bound, critical-path-limited. Algorithmic choice
  (DIT vs DIF) is the knob that matters.
- **Medium R** (=16, =32): memory-bandwidth-bound on consumer chips,
  frontend-bound on server chips. t1s and log3 help.
- **Large R** (not tested yet): register-pressure-bound likely becomes
  binding.

## Predictions for R=64

R=64 butterflies have ~64 complex inputs in flight. On AVX-512's 32 ZMM,
that's 2× over register capacity. On AVX2's 16 YMM, 4×. Spills will
dominate.

- **DIT vs DIF gap** at R=64 is a different question than R=8. The
  critical-path argument scales log2(R), so DIF might still be 10% faster
  — but spill/reload costs may dominate the arithmetic cost entirely.
- **Ladder** might finally win on R=64 AVX-512 since it's explicitly
  designed for register-constrained wide-vector codelets.
- **t1s** should still dominate at mid-large me since the memory
  bottleneck there scales with R.
- **buf** may matter on consumer chips at small me for the same DTLB
  pressure reasons.

## Raw numbers (SPR, fwd, padded ios=me+8)

| me | DIT AVX2 | DIF AVX2 | DIT AVX-512 | DIF AVX-512 |
|---|---|---|---|---|
| 64 | 239 | 218 | 143 | 130 |
| 128 | 468 | 445 | 285 | 255 |
| 256 | 852 | 693 | 479 | 434 |
| 512 | 2162 | 1925 | 1558 | 1472 |
| 1024 | 4075 | 4181 | 3316 | 3110 |
| 2048 | 9533 | 8371 | 7833 | 6822 |

Per-element costs (ns/element):

| me | DIF AVX2 | DIF AVX-512 |
|---|---|---|
| 64 | 0.426 | 0.254 |
| 256 | 0.338 | 0.212 |
| 1024 | 0.510 | 0.380 |
| 2048 | 0.511 | 0.416 |

AVX-512 ~40% faster per-element than AVX2 at R=8. This is roughly what
we'd expect: 2× vector width × 0.7 for non-perfect scaling.

## What's left to do

1. **Run on Raptor Lake** for cross-chip comparison. Prediction: DIF still
   dominates but with different per-me split. Raptor Lake has narrower
   frontend than SPR, so the front-end-bound aspect might favor DIT
   (shorter per-butterfly code) in some regimes even though DIF has
   shorter critical path.

2. **Bottleneck coverage doc update** — log3 / prefetch / tile rationale
   was R=16/R=32-focused; should add R=8 "compute-bound, algorithmic-
   choice-only" as its own category.

3. **(Lower priority) Test DIF-log3 at R=8** — current bench skipped log3
   because its signature differs from t1_dif. Would require harness work.
   Probably not worth the effort given DIF already wins decisively.
