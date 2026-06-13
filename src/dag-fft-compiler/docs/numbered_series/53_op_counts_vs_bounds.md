# 53. Op Counts vs Lundy/Van Buskirk Lower Bounds

## Question

Where do our codelets stand vs the published lower bounds for FFT op
counts? If we're at the bound, the only remaining wins are FMA-shape
and scheduling. If we're above, there's a known-better algorithm.

## Methodology

Counted SIMD instructions in each codelet, with FMA counted as 2
FLOPs (1 mult + 1 add). Each SIMD instruction is one conceptual real
operation in the algorithm (lanes batch parallel transforms, not
adding work to the operation count).

Bounds used:

- **Yavne split-radix (1968)**: classic complex-FFT op count formula
  `T(N) = 4N log₂(N) - 6N + 8`. Held as the lower bound for 39 years.
- **Johnson-Frigo modified split-radix (2007)**: current best known
  for complex FFT power-of-2 sizes,
  `T(N) ≈ (34/9) N log₂(N) - (124/27) N + lower-order`. Equivalent to
  the Lundy-Van Buskirk bound at these sizes.
- **Sorensen et al. (1987) split-radix r2c**: published lower bound
  for real-input FFT, `T(N) = 2N log₂(N) - 4N + 6`. About half of
  complex.

## C2C results: we are at the bound

```
   N    Ours   Yavne   Δ Yavne   JF/LVB   Δ JF/LVB
   8     57      56     +1.8%      56      +1.8%
  16    170     168     +1.2%     168      +1.2%
  32    470     456     +3.1%     456      +3.1%
  64   1194    1160     +2.9%    1152      +3.6%
 128   2890    2824     +2.3%    2792      +3.5%
 256   6778    6664     +1.7%    6552      +3.4%
```

**Result: we are 1-3% over the Yavne bound, 3-4% over the modern JF/LVB
bound, across the full radix range.** This is essentially at the
theoretical floor.

This is a notable validation of the algsimp + CSE work. Our codelets
use straight Cooley-Tukey decomposition, *not* split-radix — yet the
op count matches the split-radix bound. The combination of
hash-consed constant folding, factor_common_muls, share_subsums,
transpose CSE, and FMA pattern recognition is collectively eliminating
the "extra" multiplications that distinguish a naive CT decomposition
from an explicit split-radix algorithm.

The 1-3% residual gap to Yavne is essentially noise:
- Different operand ordering at the leaves
- A few FMA shape variations
- Constants that algsimp doesn't fold (e.g., -1, ±i sentinels)

**For c2c, no abstract-op wins remain.** Future gains are
FMA-fusion-shape (already at parity with hand per doc 38's findings)
and scheduling (already mature via SU/GH/BB per docs 09-13).

## R2C results: 30-90% over the bound

```
   N    Ours   Sorensen   Over    Ratio
   8     42      22       +91%   1.91x
  16    127      70       +81%   1.81x
  32    328     198       +66%   1.66x
  64    806     518       +56%   1.56x
 128   1882    1286       +46%   1.46x
 256   4283    3078       +39%   1.39x
 512   9580    7174       +34%   1.34x
```

We're significantly over the Sorensen bound at all N. The gap *shrinks*
with N: 91% at N=8 down to 34% at N=512, but it doesn't close.

## Where the r2c gap actually lives

Our r2c codelet decomposes as: `c2c(N/2)` + post-process butterfly. So:

```
ours_r2c(N) = ours_c2c(N/2) + butterfly_cost(N)
```

Diagnostic: how does `ours_c2c(N/2)` alone compare to the Sorensen
bound for full r2c(N)?

```
   N    Sorensen   ours_c2c(N/2)   excess_if_butterfly_were_free
  16       70           57                  -13   (under!)
  32      198          170                  -28
  64      518          470                  -48
 128     1286         1194                  -92
 256     3078         2890                 -188
 512     7174         6778                 -396
```

**Our inner c2c is already under the Sorensen budget.** If the
post-process butterfly were free, we'd *beat* the Sorensen bound.

So the butterfly is where the gap lives. Counting it directly:

```
   N   our_butterfly   theoretical (~6N for "monolithic" Hermitian ext)
  16        70                96
  32       158               192
  64       336               384
 128       688               768
 256      1393              1536
 512      2802              3072
```

**The butterfly itself is at theoretical complexity.** We're slightly
under the 6N textbook count (algsimp + CSE recovers some sharing). So
the butterfly implementation isn't the problem.

The problem is **structural**: our approach is `c2c(N/2) + butterfly`,
which pays the full Hermitian-extraction cost at the end. Sorensen's
algorithm interleaves Hermitian exploitation throughout the cascade,
distributing the equivalent of ~3N ops across all stages instead of
concentrating ~6N at the post-process. That's the gap.

## The known-better algorithm exists

Sorensen et al. (1987) "Real-valued fast Fourier transform algorithms"
describes the algorithm directly. FFTW implements it as
`gen_r2cf.ml` (first stage) + `gen_hc2hc.ml` (middle) + `gen_hc2c.ml`
(last stage). The intermediate data is stored in **Hermitian-packed
format** throughout the cascade, meaning each stage operates on
half-spectrum complex data and only the unique outputs are computed.

This is exactly the Stage C work flagged in doc 52. The op-count
analysis here quantifies the win:

| N | Our ops | Sorensen | Savings if Stage C lands |
|---|---------|----------|---------------------------|
| 64 | 806 | 518 | 36% |
| 128 | 1882 | 1286 | 32% |
| 256 | 4283 | 3078 | 28% |
| 512 | 9580 | 7174 | 25% |

At production sizes, 25-32% op-count savings are available via Stage C.

## How much of the op-count savings translate to runtime savings?

Important caveat. Op count is a *theoretical* lower bound; runtime
depends on FMA fusion ratio, scheduling, and memory access patterns.

Doc 50's analysis showed gcc-11 -O3 -mfma already fuses raw
`add(mul(x,y), z)` patterns into fmadd at compile time. So our 1.34x
op-count excess vs Sorensen at N=512 may translate to *less* than a
1.34x runtime excess, because gcc converts more of our adds+muls into
single FMA instructions than a "by-hand" count would suggest.

Doc 51's monolithic-vs-3-pass bench at N=128 already shows the
monolithic codelet at 1.41x faster than the 3-pass mirror with the
same algorithm. That bench measured *fused vs unfused at the same
algorithm*. Stage C would measure *fused at a better algorithm*. The
two wins are independent and stack:

- Stage A (cascade pack fusion): captures part of mono-vs-3pass win
- Stage B (cascade harness, bench): measures Stage A in isolation
- Stage C (hc2c, Hermitian-preserving cascade): captures Sorensen
  bound

If Stage A captures most of the mono-vs-3pass win and Stage C
captures the Sorensen op savings, the combined improvement over the
current 3-pass `r2c.h` is multiplicative.

## Conclusions

1. **C2C is at the bound.** Within 1-3% of Yavne, within 3-4% of LVB.
   Don't chase further op-count reduction for c2c. Optimize via FMA
   shape (already at parity with hand) and scheduling (already mature).

2. **R2C is 25-91% over Sorensen, gap shrinks with N.** This is a
   real gap with a known-better algorithm: Sorensen-style
   Hermitian-preserving cascade (FFTW `gen_hc2c`).

3. **The r2c gap is structural, not implementation-quality.** Our
   inner c2c is already under the Sorensen budget; the butterfly is
   at theoretical complexity. The gap is in algorithm choice:
   monolithic-end butterfly vs interleaved-throughout-cascade.

4. **Stage C is justified by data.** At production sizes (N ≥ 128),
   25-32% op-count reduction is on the table. Combined with Stage A
   (already built) and a real planner, this captures the FFTW
   architectural advantage in r2c.

## Recommendation

The c2c side of the project is done in terms of algorithmic op count.
The r2c side has one specific known improvement worth pursuing:
Hermitian-preserving cascade (Stage C). The size of the prize is now
quantified.

Do Stage B first anyway (build a cascade harness and benchmark Stage
A against monolithic + 3-pass). The Stage B numbers will tell you
how much of the win is pack-fusion alone — if pack-fusion alone
already captures most of the mono-vs-3pass benefit, Stage C is a
nice-to-have. If Stage A alone is unimpressive, Stage C is urgent.

The op-count analysis here makes Stage C an informed decision rather
than speculative work.
