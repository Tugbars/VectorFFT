# VectorFFT Out-of-Place Engine: Completion Report

Date: 2026-06-06. Environment: single-vCPU Cascade Lake-SP guest (KVM),
gcc-13, FFTW 3.3.10 with AVX-512, MKL 2026, single thread throughout.
All ratios from same-binary round-robin timing, min-of-rounds. All tables
follow the convention bigger number = faster.

## 1. Executive summary

VectorFFT now has a complete out-of-place execution engine in core/:
input-preserving forward and backward transforms for general N, planned
automatically from (N, K) alone, correctness-gated at machine precision,
and measured at a geomean of roughly 2.9x faster than MKL DFTI in the
matched out-of-place split configuration on this host (range 2.4 to 3.7x
across eight cells, every cell won).

The backward direction ships with zero backward codelets and zero extra
twiddle tables: the pointer-swap identity IDFT(re,im) = swap(DFT(im,re))
is applied at the dispatch layer and was validated at machine precision
at the leaf level (22 radixes), the engine level, and the full plan level.

## 2. Architecture

One plan object, three kinds, selected by rule at create time
(core/oop_plan.h, core/oop_auto.h, core/oop_execute.h,
core/oop_leaf_registry.h):

* LEAF (N <= 128): a single out-of-place codelet call. 22 leaf radixes
  {2..17, 19, 20, 25, 32, 64, 128}, coverage strictly exceeding FFTW's
  leaf set.
* BAILEY2: the primary strategy, fused four-step Bailey as a two-stage
  column-layout engine. Stage 1 is R1 long-count leaf calls with the
  transpose fused into the stores; stage 2 is one twiddled call in-place
  on dst with a K-replicated twiddle table. Natural order output
  X[k2 + R2*k1]. t1p radix set {4, 7, 8, 13, 16, 32, 64}.
* MODEB: general N through the existing stride executor. Stage 0 runs
  out-of-place via the n1 codelets' contractual OOP signature; stages 1..
  run unchanged in-place on dst. Inherits the production wisdom plans
  verbatim. Output order matches the in-place path (scrambled);
  output is BIT-IDENTICAL to the in-place dataflow.

API: vfft_oop_plan_create_auto(N, K, wisdom, hints, reg) then
vfft_oop_execute_fwd / vfft_oop_execute_bwd(plan, src_re, src_im,
dst_re, dst_im). Layout is the production column/split convention,
element e of transform t at [e*K + t].

## 3. The rule spine and the searched residue

Rules (applicability predicates, FFTW-style, all measurement-derived):

1. K must be a multiple of 8. This is the lane contract of every kind on
   this path; the proto avx512 codelets are 8-lane granular and K=4 on
   exact-size buffers overruns leg slices (found as heap corruption in
   the validation sweep, now rejected at create).
2. N <= 128 prefers the direct leaf (the two-stage pays a measured
   transposed-intermediate tax at single-codelet sizes).
3. Aliasing mask, measured boundary: a Bailey stage whose j-stride is a
   multiple of 4096 doubles (the 32KB set period) with more than 8
   streams is masked; masked cells fall to MODEB. Strides that alias only
   the 4KB L1 period are absorbed by L2 under the DFT arithmetic and
   measure fine (169/K=512, stride 52KB, runs Bailey at 2.7x vs MKL).
4. OOP requires DIT orientation (DIF would destroy the input; the same
   physics as FFTW's NO_DESTROY_INPUT predicate).
5. Backward is always the pointer swap.

Searched residue: the (R1, R2) divisor pair, nothing else. Measured pair
spread up to 24 percent across five candidates at one cell. The tuner
(vfft_oop_tune_pairs) races all unmasked pairs plus the direct leaf,
same-binary round-robin, and its winner overrides the static
balanced-first preference via a hint. The tuner corrected the static rule
twice in its first session: +10.6 percent at 1024/K=120 and a ~2 percent
leaf-beating pair at 64/K=512.

## 4. Correctness gates (all pass)

| gate | scope | result |
|---|---|---|
| leaf fwd + bwd-swap, avx512 | 22 radixes | <= 1.5e-14 |
| leaf fwd + bwd-swap, avx2 | 22 radixes, 0 zmm/k-regs | <= 1.5e-14 |
| engine fwd + bwd-swap | 13x13, 32x32 | <= 8.9e-15 |
| Mode B fwd | 6 plans incl. odd mixed-radix | BIT-IDENTICAL to in-place |
| Mode B bwd-swap | same 6 | BIT-IDENTICAL to swap dataflow |
| Mode B input preservation | same 6 | exact |
| column Bailey vs FFTW | 4 cells | <= 1e-14 |
| plan-kind gates (incl. bwd vs FFTW BACKWARD) | 8 cells + K%8 rejection | machine precision |
| wisdom-table sweep | 54 cells (5 leaf, 11 bailey, 38 modeb) | 0 failures |

## 5. Performance

THE MKL RACE (both out-of-place, split storage, column layout, one
thread, same binary):

| cell | kind | order | speed vs MKL |
|---|---|---|---|
| 64 K=512 | BAILEY2 16x4 (tuned) | natural | 3.68 |
| 128 K=512 | BAILEY2 4x32 (tuned) | natural | 3.24 |
| 169 K=512 | BAILEY2 13x13 | natural | 2.65 |
| 512 K=120 | BAILEY2 32x16 (tuned) | natural | 2.56 |
| 1024 K=120 | BAILEY2 16x64 (tuned) | natural | 2.37 |
| 1024 K=256 | BAILEY2 8x128 | natural | 2.51 |
| 4096 K=256 | MODEB | scrambled | 2.73 |
| 2310 K=32 | MODEB | scrambled | 3.51 |

Geomean ~2.9x. A second run drew 2.07 to 3.58 on overlapping cells.

FFTW context (separate sessions, guru interface, OOP both sides): parity
to 1.07x at the pow2 sizes where FFTW is strongest, 1.8x floor and 2.3 to
3.4x typical at odd sizes, plus FFTW measured 2.4 to 3.7x slower when
forced onto our split layout.

Workflow comparison for input-preserving callers (speed vs
memcpy-then-in-place = 1.000): direct OOP 0.998 to 1.162 across six
plans. Pure in-place (destroys input) runs 1.37 to 1.59, which is the
working-set price of input preservation itself, not implementation cost.

## 6. Defects found and fixed during validation

* K=4 lane-contract violation: heap corruption on exact-size buffers,
  root-caused to 8-lane granular codelets, rejected at create.
* Aliasing mask initially blunter than its own evidence (4KB period);
  refined to the measured 32KB boundary, which upgraded six sweep cells
  to natural-order Bailey and made 1024/K=256 simultaneously natural and
  faster (3.22M vs 3.57M cycles).
* Tuner pair hints could shadow a faster direct leaf at N <= 128; the
  leaf now competes as a tuner candidate.

## 7. Honesty block

1. Single noisy virtualized host; every ratio needs EPYC 9575F and
   i9-14900KF confirmation before publication. Cross-run FFTW/MKL
   absolute cycles drift; same-binary ratios are the only numbers quoted.
2. MKL's REAL_REAL split storage is its weaker configuration. The
   interleaved OOP comparison at N=1024 (May 31 session) measured 1.07 to
   1.44x. Split is our native and latency-relevant format, but any public
   claim should carry both numbers.
3. MODEB rows deliver scrambled order against MKL's natural order; a
   reorder pass is not measured.
4. MODEB runs the generic stage loop; the tier-1 plan-shaped fast path is
   not OOP-wired (a documented 5 to 6 percent left on the table).
5. v1 is single-threaded. Both ISAs are wired: the registry selects
   avx512 or avx2 per build, all plan kinds gate on both (avx2 odd-N
   Mode B excepted: the avx2 in-place set is pow2-only; odd N on avx2 is
   served by the OOP kinds). avx2 preview vs MKL on this host: 1.20 to
   2.02x with MKL free to dispatch avx512, the lower bound for the i9.
   Against FFTW the avx2 gap is OPEN: a symmetric race vs an avx2-only
   FFTW build has FFTW winning six of seven cells by 1.12 to 2.0x (we
   take 4096 MODEB at 1.09). The deficit tracks codelet bodies, not
   strategy; avx2 emission tuning is the queued work item. i9 validation
   pending.

## 8. Remaining work

Hardware audits (EPYC, i9; confirm every ratio, run the strategy
kill-list audit, recalibrate estimate constants). Tier-1 OOP wiring.
Threading (mirror the in-place K-split). Natural-order reorder option for
MODEB. avx2 t1p generation for engine parity on AVX2-only targets.
Hygiene: correction note in the May-31 conclusion doc, swap-gate
extension to spec/log3 variants.
