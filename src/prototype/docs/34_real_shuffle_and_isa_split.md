# 34. Real-Shuffle Multi-Stage Harness — Correcting Doc 33's Boundaries

## Context

[Doc 33](33_R128_R512_sweep.md) introduced the R=128 → R=512 monolithic
codelet sweep and identified what looked like a clean crossover at
R=512, B=128 — monolithic winning at small B, multi-stage winning at
large B by up to 61%. The doc flagged a caveat: the multi-stage harness
omitted **data shuffle cost** (the explicit transpose a correct CT
decomposition requires), and the multi-stage numbers were therefore a
lower bound on real cost.

This doc closes that loop. We built a real-shuffle harness that
performs a semantically correct CT multi-stage execution (verified
bit-equivalent to monolithic), measured both AVX-512 and AVX2, and
the picture changes in two ways:

- The R=512 "crossover" at B=128 on AVX-512 was largely an artifact.
  With real shuffle, the boundary moves to B=256 and presents as parity
  rather than a clear multi-stage win.
- On AVX2 the boundary is sharper and real. Multi-stage CT(16, 32) wins
  by 33% at B=512. The same N produces a much larger codelet (2.4-2.8×
  more stack ops, 22% larger object code) because of AVX2's smaller
  register file, and the OoO-resource ceilings from doc 33 kick in
  earlier.

The practical consequence: the picker needs to be (N, B, ISA)-aware,
and the boundaries need to be measured at install time on the user's
hardware rather than baked in as constants.

## The harness

A correct Cooley-Tukey decomposition of DFT-N = N1 × N2 over B batches
of complex inputs, with input/output in standard SoA layout
`re[n * B + b]` for n ∈ [0, N), b ∈ [0, B), is:

```
Pass 1: radix_N2 t1_dit with UNIT twiddles
  K = N1 * B,  ios = N1 * B
  Operates on input as if it were [n2][n1][b], doing DFT-N2 over n2.
  Output layout: pass1[k2 * N1 * B + n1 * B + b]

Transpose: data is moved from [k2][n1][b] layout to [n1][k2][b]
  buf2[n1 * N2 * B + k2 * B + b] = pass1[k2 * N1 * B + n1 * B + b]

Pass 2: radix_N1 t1_dit with W_N^{n1 * k2} twiddles baked in
  K = N2 * B,  ios = N2 * B
  Operates on buf2 as [n1][k2][b], doing DFT-N1 over n1.
  Inter-stage twiddles encoded in Pass 2's t1_dit twiddle array,
  so no separate twiddle pass is needed.
  Output layout: out[k1 * N2 * B + k2 * B + b]
  This is exactly the natural order for FFT output index
  k = k1 * N2 + k2 at flat position k * B + b.
```

The cost breakdown per call: two codelet calls (same as the "equivalent
work" harness in doc 33) plus one extra transpose pass through the
full N * B data. The transpose is scattered-write contiguous-read, which
isn't free even when the data fits in cache.

Pass 2's twiddles are W_N^{n1 * k2} for n1 ∈ [1, N1), k2 ∈ [0, N2),
replicated across the B batch dimension. The t1_dit codelet
pre-multiplies inputs by these before doing DFT-N1, which is the
standard FFTW recursive pattern.

Correctness verification: the harness runs both monolithic R=N and
multi-stage CT(N1, N2) on the same random input and compares element-wise
relative error. Results agree to ~1e-13 across all tested configurations,
on both ISAs.

```
AVX-512:
  CT(8, 16) of R=128:   max_rel = 1.29e-13   PASS
  CT(16, 16) of R=256:  max_rel = 3.15e-13   PASS
  CT(16, 32) of R=512:  max_rel = 6.90e-13   PASS

AVX2:
  CT(8, 16) of R=128:   max_rel = 1.29e-13   PASS
  CT(16, 16) of R=256:  max_rel = 3.15e-13   PASS
  CT(16, 32) of R=512:  max_rel = 6.66e-13   PASS
```

## What changed from doc 33's "equivalent work" harness

The harness in doc 33 called Pass 2 with `ios = N2 * B` (contiguous
access). This implicitly assumed data was already in the layout Pass 2
needed — i.e., that a free shuffle had happened. By skipping the
transpose, the harness gave Pass 2 cache-friendly access on data that
in reality requires rearrangement first.

The honest harness includes the transpose. Pass 2 still runs with
contiguous access (same `ios = N2 * B`), but Pass 1's output goes
through `buf2[n1 * N2 * B + k2 * B + b]` rather than being read in place.
The cost difference is one full read+write pass over N * B complex
elements (~512 KB at N=128, B=128).

## AVX-512 with real shuffle

### R=128, CT(8, 16)

```
B      mono     realshuffle  rs/mono   doc 33 (simple)
8      1110     1029         0.93      0.82  (multi narrower, still wins)
16     2431     3619         1.49      1.14  (mono advantage grew)
32     5796     8588         1.48      1.31
64     12408    16994        1.37      1.39
128    21382    35305        1.65      1.74
256    45879    77224        1.68      1.64
512    127793   222268       1.74      1.31  (mono advantage GREW)
1024   383594   508713       1.33      1.43
```

Times in ns per DFT-128 batch. The shuffle cost shifted multi-stage
from "competitive at large B" to "decisively worse" at large B. The
small-B multi-stage win persists.

### R=256, CT(16, 16)

```
B      mono     realshuffle  rs/mono   doc 33 (simple)
8      2907     2746         0.95      0.84
16     6170     7821         1.27      1.20
32     12208    16311        1.34      1.25
64     22916    30534        1.33      1.32
128    46924    68452        1.46      1.51
256    102501   211066       2.06      1.32  (mono +106% with shuffle)
512    271481   483502       1.78      1.12  (mono +78% with shuffle)
```

The flattening at large B that doc 33 noted as evidence the crossover
was approaching was the shuffle-cost mirage. With honest accounting,
the mono advantage grows or stays steady — it doesn't collapse.

### R=512, CT(16, 32) — the corrected picture

```
B      mono     realshuffle  rs/mono   doc 33 (simple, CT 16x32)
8      8071     9935         1.23      1.12
16     17392    19960        1.15      1.10
32     35097    39540        1.13      1.22
64     71486    86751        1.21      1.09
128    199875   258917       1.30      0.89  (was multi +12%, now mono +30%)
256    570976   577674       1.01      0.67  (was multi +49%, now PARITY)
512    1227128  1217202      0.99      0.66  (was multi +51%, now PARITY)
```

The most dramatic correction. Doc 33's "crossover at B=128 with
multi-stage winning by 11-61%" was almost entirely shuffle-cost
underestimation. With real shuffle:

- B=128: monolithic wins by 30% (was: multi won by 12%)
- B=256, B=512: parity within ~1% (was: multi won by 49-61%)

There is no clear multi-stage win on AVX-512 at R=512 across the
tested batch sizes. At B ≥ 256 the two paths converge to within
measurement noise.

### R=512, CT(32, 16) — the "winner" that wasn't

```
B      mono     realshuffle  rs/mono   doc 33 (simple)
8      7487     8522         1.14      1.11
16     15289    17501        1.15      1.09
32     30057    35962        1.20      1.22
64     58638    77835        1.33      1.09
128    144543   244218       1.69      0.94  (was multi +6%, now mono +69%)
256    354139   507957       1.43      0.62  (was multi +61%, now mono +43%)
512    848441   1166503      1.38      0.62  (was multi +61%, now mono +38%)
```

CT(32, 16) was the empirical winner from doc 33's simple harness — it
won by 38-61% at large B. With real shuffle it loses to monolithic
by 38-69% across the same range.

The mechanism: CT(32, 16) puts the heavier R=32 codelet on Pass 2 at
the smaller stride (16 * B) while R=16 runs Pass 1 at the larger stride
(32 * B). The explicit transpose moves data between passes; doing dense
R=32 butterflies on data being scattered to/from cache lines hits
cache hard. CT(16, 32) reverses this — R=32 runs at the smaller stride
during Pass 1, and the post-transpose Pass 2 is the lighter R=16. The
factorization that looked best in the simple harness was the worst in
the honest one.

## AVX2 with real shuffle — same harness, different ISA

### Codelet asm characteristics

The AVX2 codelets are substantially heavier than their AVX-512
counterparts:

```
                  AVX-512                AVX2
R=N    asm-lines  FP-instr  stack    asm-lines  FP-instr  stack  AVX2/512 stack
16     ~600       211       12        708       313       110    9.2×
32     ~1500      591       84        1755      878       372    4.4×
128    ~6300      3530      689       7440      4805      1940   2.8×
256    ~14700     8541      2099      17062     11793     5266   2.5×
512    ~31700     19590     5141      37802     27052     12192  2.4×
```

The reason is the register file. AVX-512 has 32 ZMM registers; AVX2
has 16 YMM. With half the register space, the spill recipe is forced
to push roughly 2.4-2.8× more values to stack to keep computation
alive. The non-FMA work also rises (4805 vs 3530 FP-instr at R=128)
because FMA-pairing opportunities are limited by register pressure
and more pure adds/subs are needed.

Object size at R=512: 345 KB on AVX2 vs 282 KB on AVX-512 (22% larger).
L1 instruction cache is 32 KB on Skylake-X regardless of ISA, so the
icache pressure ratio is proportionally worse on AVX2.

### R=128, CT(8, 16) on AVX2

```
B       mono     realshuffle  rs/mono
8       2502     2110         0.84   (multi wins +18%)
16      6066     5755         0.95   (multi wins +5%)
32      13050    25783        1.98   (mono wins +98%)
64      25721    43049        1.67
128     62546    88468        1.41
256     102657   209778       2.04
512     391305   465668       1.19
1024    941321   1146236      1.22
```

Multi-stage wins at B=8 *and* B=16 (vs only B=8 on AVX-512). The
transition is sharper: from 0.95 at B=16 to 1.98 at B=32. Mono wins
decisively from B=32 onwards.

### R=256, CT(16, 16) on AVX2

```
B       mono     realshuffle  rs/mono
8       6878     5282         0.77   (multi wins +30%)
16      13013    16684        1.28
32      27204    43407        1.60
64      53167    86797        1.63
128     126801   207535       1.64
256     250869   439223       1.75
512     683107   931303       1.36
```

The B=8 multi-stage win is much larger on AVX2 (+30% vs +5% on AVX-512).
Above that, mono wins for all B by 28-75%.

### R=512, CT(16, 32) on AVX2 — the real crossover

```
B       mono     realshuffle  rs/mono
8       18788    21255        1.13
16      37535    46997        1.25
32      73987    107655       1.45
64      140209   241396       1.72
128     405968   521763       1.29
256     1222469  1209621      0.99   (parity)
512     3076638  2319822      0.75   (multi wins +33%)
```

This is the genuine crossover that doc 33 thought it had found on
AVX-512. On AVX2 it really exists: B=256 is parity, B=512 multi-stage
wins by 33%. The same R=512 codelet that's a marginal monolithic win
on AVX-512 is a clear monolithic loss on AVX2 at B=512.

CT(32, 16) on AVX2 loses across all B by 7-53%, same shape as AVX-512.

## Side-by-side summary

```
                R=128 CT(8,16)       R=256 CT(16,16)      R=512 CT(16,32)
B    AVX-512  AVX2          AVX-512  AVX2         AVX-512  AVX2
8     0.93    0.84          0.95     0.77          1.23     1.13
16    1.49    0.95          1.27     1.28          1.15     1.25
32    1.48    1.98          1.34     1.60          1.13     1.46
64    1.37    1.67          1.33     1.63          1.21     1.72
128   1.65    1.41          1.46     1.64          1.30     1.29
256   1.68    2.04          2.06     1.75          1.01     0.99
512   1.74    1.19          1.78     1.36          0.99     0.75
1024  1.33    1.22          —        —              —        —
```

(rs/mono ratios — values < 1 mean multi-stage wins.)

## The mechanism generalizes — boundary shifts, structure doesn't

The OoO-resource exhaustion story from doc 33 holds on both ISAs but
the thresholds where ceilings get hit are ISA-dependent:

- **L1 I-cache (32 KB on SKX, 64 KB on SPR):** R=128 fits on both
  ISAs. R=256 marginal on AVX2 (149 KB). R=512 overflows on both,
  worse on AVX2 (345 KB vs 282 KB).
- **Store buffer (56 entries on SKX, 72 on SPR):** R=512 AVX2 has
  12192 stack ops in body; with ~150-instruction OoO visibility,
  ~90 stores in flight at peak. Hard overflow. R=512 AVX-512 has
  ~75 stores in flight — marginal overflow on SKX, OK on SPR.
- **µop cache (~1.5K µops):** R=128 fits on both ISAs. R=256 overflows
  on AVX2 (FP-instr count 11793 → roughly 11793 µops). R=512 overflows
  on both.
- **Register file (168 FP PRF on SKX):** AVX2 needs more PRF entries
  per unit of computation because the recipe spills more aggressively.

The crossover where multi-stage starts winning happens when the codelet
outgrows enough of these resources simultaneously that the front-end
and store buffer stall faster than the OoO engine can hide. On
AVX-512 this happens at R=1024 (predicted, untested). On AVX2 it
happens at R=512.

## Picker design implications

The recommendation from doc 33 ("R=512 needs (N, B)-aware selection
with multi-stage above B=128") was based on the simple harness. The
real picture:

| N | AVX-512 | AVX2 |
|---|---|---|
| 128 | Monolithic for B ≥ 16 | Monolithic for B ≥ 32 |
| 256 | Monolithic for B ≥ 16 | Monolithic for B ≥ 16 |
| 512 | Monolithic for B ≤ 128, parity at B ≥ 256 | Monolithic for B ≤ 128, multi-stage CT(16, 32) for B ≥ 256 |

The picker needs three dimensions: N, B, ISA. Not the simpler
(N → factorization) decision that worked for R ≤ 64. The lookup is
still small (~24 cells for the sizes we currently support across
AVX-512 / AVX2 and roughly four B-buckets) so the planner layer is
thin. The work is in populating the table.

## These numbers are not portable — measure at install time

Everything above was measured on one VM with one Skylake-X/SPR-class
CPU and one GCC version. The thresholds will shift on different
hardware:

- Ice Lake / Tiger Lake: different store buffer depth (72 entries
  vs 56), different µop cache geometry
- Sapphire Rapids: larger L1 instruction cache (64 KB vs 32 KB) —
  R=256 may fit cleanly, R=512 boundary may shift
- Zen 4: completely different µop cache structure, different L1
  organization
- Older Skylake without AVX-512: only AVX2 path applicable
- Future CPUs with wider OoO windows: monolithic boundary may push
  further out

Hard-coding the (N, B, ISA) thresholds we found here would ship wrong
defaults on most hardware. The right design is an install-time
**wisdom-generation harness** modeled on FFTW's wisdom: run the
real-shuffle bench on the user's actual hardware, measure the
crossovers across the (N, B, ISA, factorization) grid, write a wisdom
file the planner consults at runtime.

The planner layer itself is thin: given (N, B, ISA), look up the
factorization in the wisdom file, return it. All the complexity is in
the wisdom-generation harness. Most of that harness already exists
from this session — needs to be packaged into an installable
benchmark with persistent output.

## What about R=1024?

Not measured here. The trend predicts:

- AVX-512 R=1024: monolithic codelet has ~40000 FP-instr,
  ~10000-12000 stack ops. Object ~600 KB. Crossover probably at
  B ≥ 128 with clear multi-stage win.
- AVX2 R=1024: codelet has ~60000-70000 FP-instr, ~25000-30000 stack
  ops. Object ~700+ KB. Multi-stage likely wins from B ≥ 32 or so.

R=1024 is also where the simple harness fails to test before its
shortcomings would have been obvious — at this size, the shuffle cost
relative to compute is still significant but the OoO ceiling dominates.
Worth measuring once the wisdom-generation harness is in place since
it'd already be running the right grid.

## Where this leaves the open work

Reordering doc 33's open-work list in light of these findings:

The first is the install-time wisdom-generation harness. Builds on what
this session produced. The (N, B, ISA, factorization) grid is well-defined.
Outputs a wisdom file the planner consults. Shipping this enables the
correct planner behavior on arbitrary hardware without us needing to
characterize every CPU.

The second is the spill-controller redesign. The AVX2 data makes this
more compelling than doc 33 anticipated. R=512 AVX2's 12192 stack ops
is the dominant cost; reducing to ~8000 would likely shift the AVX2
crossover from B=256 to B=512 or eliminate it. AVX-512's 5141 stack
ops at R=512 also has room — reducing to 3500-4000 might make R=512
monolithic the clear winner at B=256+.

The third is R=1024 measurement (both ISAs) to define the next
monolithic boundary. Quick once the harness is packaged.

The fourth, separately, is cross-stage codelet fusion — still relevant
as an attack on the 21% executor overhead from the VTune analysis.
Independent of where monolithic scales to. Multi-week investment.
