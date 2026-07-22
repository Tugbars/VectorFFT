# Row-major executor & codelets — closing the row-major K gap

Status: roadmap (not started). Companion docs: `arbitrary_k_vectorization.md`
(K-axis tail story), `../performance/mkl_geometry_contracts.md` §6a16–6a17
(boundary-fold pattern this plan reuses).

## 1. Measured gaps and their decomposition

Two diagnostic numbers from prior layout benches (user machine, same-process):

| case | vs MKL | what it measures |
|---|---|---|
| K=1, any layout | 0.19× | pure axis collapse — at K=1 row-major and lane-major coincide (one contiguous transform), so no layout tax exists; slice_K=1 drives every codelet into its scalar tail: no SIMD, no ILP, full per-call overhead per element |
| row-major, K=4 | 0.69× | three stacked factors, decomposed below |

Row-major K=4 factor stack:

| factor | size | mechanism |
|---|---|---|
| axis-length penalty | mild | K=4 fills exactly one YMM — vectorized, but zero k-block ILP, per-call/per-group overhead amortized over only 4 lanes |
| layout tax | real | either two O(NK) transpose sweeps (row↔lane) or stride-N lane gathers inside every codelet load |
| opponent upgrade | real | row-major batched (`DFTI_INPUT_DISTANCE = N`, contiguous transforms) is MKL's HOME layout — the corner they have tuned longest. Same lesson as the (100,4) MKL-native-IL discovery: change the layout contract, change the opponent |

0.69× on the opponent's most-tuned corner is not an embarrassing baseline; it is
what "wrong layout, their turf" costs. K=1 and row-major-K≥4 need DIFFERENT
fixes — K=1 needs four-step self-batching (see §7), row-major K≥4 is a
boundary-fold + tail-codelet problem, i.e. mostly machinery we already have.

**Diagnostic to run before any codegen:** same-process, lane-major K=4 vs
row-major K=4 vs MKL. MKL held constant, the ratio between OUR two arms
isolates the pure layout tax. If lane-major K=4 runs ~1.2–1.5× MKL (consistent
with the v1.0 sweep), the layout tax is ~1.7–2.2× — fully explained by two
extra memory passes at thin-compute small N.

## 2. The architectural fact that makes this cheap

**The t1 codelet ABI already accepts per-element twiddle arrays.** The
kb-broadcast machinery passes `tw_re/tw_im[j][k]` arrays; the generic executor
happens to fill them with broadcast constants, but the codelet body neither
knows nor cares. Feed genuine per-element tables and the existing butterfly
lattice computes an intra-transform-vectorized stage UNCHANGED.

## 3. Row-major geometry = lane-major lattice with slice_K := stride

Within one row-major transform, a DIT/DIF stage with stride `s` has butterflies
at contiguous instance index c ∈ [0, s), legs `s` apart:

    leg j of butterfly c  →  base + j*s + c        (unit-stride along c)

That is EXACTLY the lane-major load lattice with `slice_K := s`. Vectorize
along c: same codelets, same lattice — only the twiddles now vary along the
vector axis (per-element tables instead of broadcasts), and the K rows of the
batch amortize each stage's tables K×.

## 4. Scope: what exists vs what is genuinely new

| piece | status | work |
|---|---|---|
| interior stages (stride ≥ VLEN) | **exists** — t1 codelets driven by per-element tw tables | reuse; zero codegen |
| row-major executor | new but thin | stage walker reinterpreting the plan's group/stride structure per row + twiddle-table builder + K-row batching loop |
| tail stages (stride < VLEN) | **genuinely new codelet family** | the crux — §5 |
| rm boundary folds (complex-interleaved row ↔ split) | trivial new family | contiguous deint/int, pure shuffles, no strided lattice — the easy cousin of il_in/il_out; same generator + gate methodology as 6a16 |
| order contract | decision | scrambled-out stays free; natural-order-out costs the reorder MKL also pays — price it explicitly in benches |
| extra twiddle-load stream | cost to watch | per-element tables are ~N doubles per twiddled stage — cache-resident at small N, but a real load stream the broadcast path never had; the bound prototype measures whether it eats the win |

## 5. Tail stages: the crux, not a corner case

A stride=1 stage exists in EVERY plan (last DIT stage); for our 2-stage
large-radix plans that is **half the compute**. Along c it is unvectorizable.
Classic resolutions:

1. **Shuffle-network R-point codelets** — FFTW genfft territory; full parity
   possible, months of generator work, their home turf.
2. **Axis-switch codelets (preferred)** — when stride drops below VLEN, a fused
   in-register transpose flips the vector axis from c to k, and the remaining
   stages run k-vectorized with broadcast twiddles like home. K=4 on AVX2 is
   the sweet spot: the 4×4 double transpose is the standard
   unpcklpd/unpckhpd + perm2f128 block (~2 shuffles/element), and the butterfly
   bodies are the EXISTING k-vectorized ones with a new load/store lattice —
   one generator flag family (working name `--rm-tail`), not a new theory.

## 6. Phased plan

| phase | content | gate/criterion |
|---|---|---|
| 0 | decomposition bench (lane-major K4 vs rm K4 vs MKL, same-process) | confirms tax split; no code |
| 1 | bound prototype: existing t1 codelets driven c-wise via real tw tables + naive scalar tail + contiguous deint/int sweeps | one afternoon, no codegen. Number tells (a) what fraction of 0.69→1.0 the tail family must carry, (b) whether the tw-load stream hurts |
| 2 | if bound ≥ ~0.85×: generate axis-switch tail family + rm boundary folds; bit-gate vs generic reference per 6a16 methodology | ALL BIT |
| 3 | executor productization: plan-time rm path selection, wisdom column if plans differ, public layout flag (shared with the deferred vfft_config item) | bench table in geometry contracts |

## 7. Relation to the K=1 problem (different plan)

K=1 cannot be fixed by rm folds — there is no batch axis at all. The plan there
is **four-step self-batching**: N = N1×N2, pass 1 = N1-point FFTs batched
K=N2 (native layout!), twiddle pass, blocked transpose, pass 2 native again.
Reuses everything; new parts are the blocked transpose kernel (already owed to
the large-K TLB/Bailey backlog) + twiddle pass + plan glue. Output is
transposed/digit-reversed — natural order costs the back-transpose; include it
in honest benches. Expected landing: 0.19× → ~0.6–1.0×, unmeasured.

## 8. Honest expectations

c-vectorized interior + tail resolution is precisely FFTW's architecture, and
MKL still usually edges FFTW in this corner. Target for row-major K≥4:
**0.69× → 0.9–1.1×**, with the scrambled-order discount as the remaining
structural edge where the consumer accepts it (convolution-class work). Full
parity at natural-order row-major means re-fighting a twenty-year war — decide
per market need, not pride.
