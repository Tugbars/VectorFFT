# VectorFFT OOP Strategy: Fused Four-Step Bailey

Date: 2026-06-06. Status: settled by controlled experiment. This document
records the decision, the evidence that forced it, the mechanisms behind
it, and the falsification plan. All numbers from same-binary round-robin
timing on a single Cascade Lake-SP guest; tables are bigger = faster.

## 1. The decision

The primary out-of-place strategy is fused four-step Bailey with the
fattest possible two stages: stage 1 performs the column DFTs with the
transpose fused into its store pattern, stage 2 performs twiddle multiply
and row DFTs as one in-place call on the destination. Recursion only when
a factor exceeds cache or codelet size. Stockham, deep factorization, and
six-step are not runtime strategies; they survive only as audit harnesses.

The strategy axis is a rule. The only searched quantity is the divisor
pair (and codelet variant), ranked by a small measured tuner and cached.

## 2. The taxonomy, so the names mean the same thing

Bailey 1990 defines both relatives. Four-step: column FFTs, twiddle,
transpose, row FFTs. Six-step: transpose, FFTs, twiddle, transpose, FFTs,
transpose, three explicit data passes so every FFT touches contiguous
rows; designed for external memory and vector machines where strided
access was ruinous. Our variant names:

* A, fused Bailey: four-step collapsed to its minimum. The transpose
  never exists as a pass; it is the stage-1 store pattern. The twiddle
  never exists as a pass; it is fused into the stage-2 codelet.
* F, ping-pong Bailey: same, but stage 2 reads a work buffer instead of
  running in-place on dst.
* B, Stockham 2-stage: stage 1 stores naturally, stage 2 absorbs the
  permutation in its loads.
* E, six-step direction: one explicit transpose pass inserted.
* Deep CT: more than two thin stages.
* Recursive (FFTW's method): at two stages it produces the identical
  plan to A; the distinction only exists at depth.

## 3. The evidence

### 3.1 Strategy isolation, identical codelets, only the wrapping varies

Eleven cells, N = 49 to 8192, every variant correctness-gated <= 1e-14,
relative speed with A = 1.000:

| cell | N | A fused | F ping-pong | B Stockham | E six-step |
|---|---|---|---|---|---|
| 7x7 | 49 | 1.000 | 0.781 | 0.759 | 0.704 |
| 8x8 | 64 | 1.000 | 0.769 | 0.742 | 0.657 |
| 13x13 | 169 | 1.000 | 0.866 | 0.832 | 0.745 |
| 16x16 | 256 | 1.000 | 0.814 | 0.771 | 0.711 |
| 16x32 | 512 | 1.000 | 0.901 | 0.873 | 0.733 |
| 32x16 | 512 | 1.000 | 0.864 | 0.791 | 0.695 |
| 32x32 | 1024 | 1.000 | 0.925 | 0.858 | 0.706 |
| 32x64 | 2048 | 1.000 | 0.925 | 0.814 | 0.686 |
| 64x32 | 2048 | 1.000 | 0.880 | 0.814 | 0.657 |
| 64x64 | 4096 | 1.000 | 0.966 | 0.895 | 0.772 |
| 64x128 | 8192 | 1.000 | 0.907 | 0.861 | 0.718 |

A wins every cell. Memory-bound spot checks at 4x K amplify the ordering
(for example 64x64: F 0.843, B 0.689, E 0.672). The wrong dataflow costs
up to 1.6x with identical arithmetic.

### 3.2 The decomposed mechanisms (each isolated by a variant pair)

* In-place second stage beats ping-pong by 8 to 16 percent: dst is hot,
  and the working set carries one less buffer.
* Store-side permutation absorption beats load-side by 4 to 10 percent:
  stores drain in the background, strided loads sit on the DFT dependency
  chain.
* Every extra full pass over the data costs 25 to 50 percent, more when
  memory-bound. Passes bill at DRAM prices.

### 3.3 Deep factorization

engine_natural_oop_4stage (4x4x4x16) is the standing confirming negative:
deep plans fragment into many small strided calls and lose despite better
numerical error. Shallowest factorization is a rule.

### 3.4 The big-N cell where six-step was supposed to win

N = 64^3 = 262144, ~100MB working set, far past L3. A3 = fully fused
three-pass Bailey, all loads 64-stream at 256KB stride. E6 = six-step
shape, two blocked transposes plus three contiguous passes. Pre-registered
prediction: E6 wins on load aliasing. Result: A3 = 1.000, E6 = 0.77 to
0.79. The prediction failed because L3 slice-hashing scrambles the strided
streams across slices and each 64-load group hides behind ~950 FP
instructions, while transposes are pure traffic with nothing to hide
behind. Pass count rules even harder out of cache.

Structural finding from the same derivation: with affine strides a fused
three-stage cannot produce natural order (digit-permutation parity), so
six-step's transposes buy ORDERING, not speed. Its remaining speed domain
is where the transpose is forced anyway: external and distributed memory,
which is exactly where FFTW hand-wrote it (the MPI layer) and nowhere
else; verified against FFTW 3.3.10 source, whose single-node solver
inventory contains no Stockham, four-step, or six-step rule at all.

## 4. How the planner uses this

FFTW-shaped, deliberately simple, no offline-tuning superstructure:

* Rules as applicability predicates: leaf below 129 direct; OOP implies
  DIT; K multiple of 8 (the avx512 lane contract, enforced after a real
  heap-corruption find); backward is the pointer swap; the aliasing mask.
* The aliasing mask carries its own measurement history: a Bailey stage
  j-stride that is a multiple of 4096 doubles (32KB set period) with more
  than 8 streams is masked, and masked cells fall to Mode B, whose wisdom
  factorizations use radixes that fit associativity. The first draft
  masked at the 4KB period and was provably too blunt: stride 52KB
  (169/K=512) aliases L1 only, L2 absorbs it under the arithmetic, and
  the cell runs Bailey at 2.7x vs MKL once unmasked.
* The searched residue is the divisor pair, measured by a same-binary
  tuner that includes the direct leaf as a candidate. Measured pair
  spread reaches 24 percent; the tuner overrode the static preference
  twice in its first session. No universal big-leaf or big-twiddle rule
  exists; section-14 data shows the winning order flipping between sizes.
* Wisdom caches the winners; the estimate tables (calibrated per-codelet
  CPE and memory-boundedness, already the production mechanism) remain
  the cold-start fallback.

Layout note: the raced engines were lane-blocked (element stride one
cache line), which is immune to the aliasing mask by construction and
remains the benchmark champion. The shipped API uses the production
column/split layout for compatibility; the mask exists because column
layout re-exposes power-of-two stride aliasing. A lane-blocked buffer API
for batch-heavy callers would recover the raced numbers and is a known
option, not a commitment.

## 5. Falsification stance

Every kill on the list is one host's measurement with a mechanism story,
not a theorem. The audit harnesses (strategy isolation, the rxr sweep,
the big-N cell) stay in benchmarks/ so any new microarchitecture gets one
audit pass before its wisdom is blessed. The exposed edges, in honesty
order: very large single-transform N beyond what this container holds
(the regime rule extends past 100MB here, the crossover is unmeasured),
Zen 5 cache behavior, and the untuned transposes in the six-step
harness. If an audit ever flips a rule, that is a finding, and the
planner gains a predicate, not a search.
