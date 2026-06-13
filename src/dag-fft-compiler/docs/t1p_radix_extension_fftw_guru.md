# t1p radix extension and the FFTW guru OOP races

Short version: the OOP codelet generator extends to new t1p radixes (7, 8, 13
emit, compile, and verify at 5e-15 with zero compiler changes). In OOP-vs-OOP
guru races on N = R x R two-stage engines, VectorFFT beats FFTW everywhere FFTW
must run its own twiddle machinery, and the optimization flags turn out to be
strongly context-dependent: compile-time strides (spec) are a 16-30 percent win
on small radixes and neutral-to-negative on the radix-64 leaf, while
fuse+store-fused alone is neutral-to-negative almost everywhere despite halving
spills. All conclusions below are from round-interleaved single-binary races
unless flagged otherwise.

## 1. Environment and method

Single-vCPU KVM guest (Cascade Lake-SP class), gcc-13, FFTW 3.3.10 built with
--enable-avx512 --enable-fma (588 avx512 symbols in libfftw3.a). Cross-run
absolute cycles drift up to 1.5x (deschedule spikes), so the only trusted
numbers come from one binary timing all contenders in interleaved rounds
(ROUNDS=40, per-contender min, FTZ/DAZ on, warmup first). Cross-binary
comparisons are flagged where they appear. FFTW plans are FFTW_PATIENT via the
guru64 interface, out-of-place, plan strings recorded. Correctness gate: every
engine variant validated against fftw_plan_dft_1d, threshold 1e-9, all passed
at 5e-15 to 9e-15.

Layout: K transforms lane-blocked in groups of V=8, split re/im, element e of
lane l at [blk*N*V + e*V + l]. Engines are natural-order, input-preserving,
two-stage: n1_oop column stage then t1p row stage called in-place on dst
(strides generalized from the validated 32x32 engine, with 32 replaced by R).

## 2. Generation

Canonical flags (from codelets/regen_spec_r32.sh):

    gen_radix R --oop --oop-buffer-oop --oop-load UG --oop-store UG \
      --isa avx512 --emit-c [--twiddled-pos] [--log3] \
      [--fuse 8 --oop-store-fused] [--oop-strides L,G,OL,OG]

All of r7/r8/r13 x {n1_oop, t1p, t1p_log3} x {base, opt, spec} emit and compile
clean. The Winograd prime path works through the OOP emitter unchanged. Codelet
ABI (generic): 11 args (src re/im, dst re/im, Wr/Wi, L, G, OL, OG, count). Spec
ABI: 7 args (strides baked). Twiddle table: Q[(l2-1)*R + k2] = exp(-2*pi*i*
l2*k2/N), l2 in 1..R-1.

## 3. Variant races (interleaved, two runs each, robust)

N=49 (7x7, K=2048) and N=169 (13x13, K=1024), flat t1p unless noted:

| variant | r7 vs base | r13 vs base |
|---|---|---|
| base (first cut) | 1.000 | 1.000 |
| log3 | 0.96 / 0.98 | 0.86 / 0.90 |
| +fuse 8 +store-fused (opt) | 1.04 / 1.05 (worse) | 0.89 / 0.95 |
| opt + baked strides (spec) | **0.70 / 0.72** | **0.81 / 0.84** |

vs FFTW guru in the same runs: r7 spec 2.32-2.36x, base 1.65-1.67x; r13 spec
2.89-3.41x, base 2.33-2.88x. FFTW's plan: r7 = t1fuv_7_avx2 + n1fv_7_avx2_128
(its own two-stage twiddle CT, the like-for-like race); r13 = dftw-generic
wrapping n1fv_13_avx2 (FFTW ships no t1_13 twiddle codelet at all). Caveat:
FFTW's absolute cycles drift ~20-25 percent across runs while our engines stay
within ~4 percent, so the vs-FFTW ratios should be quoted as ranges; an earlier
cross-binary run with first-cut codelets saw 1.32x (r7) and 1.59x (r13) against
a faster FFTW sample. The variant deltas in the table are within-run and solid.

Radix-64 OOP leaf (N=64 direct, K=2048, interleaved, two runs):

| variant | vs base | spills/reloads (static) |
|---|---|---|
| base generic | 1.000 | 233 / 254 |
| +fuse+store-fused | 1.09 / 1.10 (worse) | 129 / 145 |
| +spec strides | 1.05 / 1.07 (worse) | 128 / 141 |

Base generic leaf vs FFTW's direct n2fv_64_avx2_128: 1.01x / 0.98x, parity.
An earlier cross-binary measurement suggesting an 11 percent loss was
deschedule noise.

## 4. The per-context flag rule

* Small radixes (7, 13): bake the strides. The codelet body is small, so the
  UG index arithmetic is a large fraction of runtime; spec deletes it for a
  16-30 percent win. log3 helps more as the twiddle count grows (r13 > r7).
* Large leaf (64): ship the generic codelet. Spills (233 static) are fully
  hidden behind compute; halving them recovers nothing and the fuse/store-fused
  schedule perturbation costs ~10 percent. Same lesson as the 2026-05-30
  in-place finding: schedule shape rules, spill count does not.
* The 32x32 engine previously measured spec as ~19 percent faster than generic,
  but that comparison was cross-binary and needs an interleaved confirm before
  being banked.

Op-count audit: the OOP r64 leaf's arithmetic is identical to the production
in-place r64 n1 (336 FMA, 50 mul, 592 add/sub), so the op-count reduction
machinery is already fully applied to OOP leaves; nothing is missing there.

## 5. M-fence / M-on status for OOP codelets

The OOP emitter (codelet_oop.ml) applies its own two-rule policy: fence ON by
default (fence-only emission, no pin), pin structurally OFF in Tier B "regalloc
deferred to Tier C". The VFFT_PIN_FORCE / VFFT_NO_REGALLOC escape hatches are
silently ignored on the OOP path because install_alloc is never called.
Wiring pin in is real compiler work, and the measured data argues its expected
value is low for these leaves: pin's win mechanism is preventing hot-value
evictions, and the evictions here are measurably free.

## 6. Structural findings vs FFTW

* FFTW's only OOP codelet family is the notw leaf (n1); every twiddle stage is
  in-place by framework construction (dftwapply takes a single pointer pair).
  General-N OOP therefore needs only n1_oop leaves; t1p remains a per-engine
  weapon for the one-call natural-order design.
* The t1p idea descends from FFTW's t-codelet (per-position twiddle walk inside
  one call); the OOP binding plus split layout, spec strides, and
  store-on-compute are the VectorFFT additions.
* Backward codelets may be unnecessary: FFTW ships zero scalar bwd codelets and
  swaps re/im pointers at plan creation (EXTRACT_REIM). The same identity
  applies even more cleanly to our split layout; pending one numerical
  validation, the _bwd half of the OOP matrix can be dropped.
* At single-codelet sizes (N=64), a balanced two-stage plan loses 15-20 percent
  to a direct leaf (the transposed intermediate is pure data-movement cost);
  the planner should prefer direct leaves there, as FFTW's does.
* FFTW forced onto our lane-blocked split layout (guru split, rank-2 howmany)
  runs 2.4-3.7x slower than our engines at every size tested: the layout moat.
* FFTW's PATIENT planner consistently selected avx2/avx2_128 genus codelets for
  small N on this machine despite avx512 being available, only using
  t1fv_32_avx512 at N=1024.

## 7. Interleaved confirms: r32 spec and the spec+log3 stack

The r32 spec-vs-generic comparison is now confirmed in a single-binary
interleaved race (bench_r32_spec_interleaved.c, four engine variants + FFTW
guru, ROUNDS=30):

| K | specF/genF | specL3/genL3 | best variant | best vs in-run FFTW |
|---|---|---|---|---|
| 128 (cache-resident) | 0.827 | 0.837 | specL3, 733,287 cyc | 1.069x |
| 512 (clean run) | 0.755 | 0.774 | specL3, 3,205,663 cyc | 2.293x |
| 512 (throttled run) | 0.917 | 1.005 | compressed | 1.89x |

Spec is a 17 percent win cache-resident and ~24 percent in a clean K=512 run;
under a whole-run throttle (bandwidth-bound) the variant differences compress
toward zero, which is expected and does not contradict the clean runs. The
earlier cross-binary "+19 percent" estimate is hereby banked.

FFTW plan variance at N=1024 deserves its own line: across guru PATIENT
invocations on this VM, FFTW produced the 32x32 plan (t1fv_32_avx512 +
n2fv_32_avx2, 3.34M cycles), a 4x(32x8) three-stage plan (t1fv_4 + t1fv_32 +
n1fv_8, 7.35M), and a throttled 12.9M sample. Honest vs-FFTW claim at N=1024
K=512: specL3 at 3.21M cycles is ~1.04x against FFTW's best observed sample,
~1.2x against its typical plan_many outcome, and 2.3x against its bad PATIENT
draws. The planner instability is FFTW's own measurement noise problem on this
host, but the conservative number is the 1.04-1.2x range.

spec+log3 stacking (5-variant interleaved races, two runs each):

| radix | specL3/spec | specL3/base | specL3 vs in-run FFTW |
|---|---|---|---|
| 7 | 0.992 / 0.989 | 0.729 / 0.720 | 2.35x / 2.70x |
| 13 | 0.945 / 0.967 | 0.694 / 0.788 | 3.39x / 3.32x |
| 32 | 0.97-0.98 (from sL/sF) | n/a (no first-cut r32) | see table above |

The stack is real where twiddle-broadcast count is high: at r13 log3 adds
3.5-5.5 percent on top of spec (12 broadcasts per position to thin), at r7 it
adds ~1 percent (6 broadcasts, nothing to save). Production default: spec
strides plus log3 for t1p stages, never worse and up to 5 percent better than
spec flat.

## 8. The full n1_oop leaf matrix and the backward-swap validation

The complete OOP leaf set is generated, compiled, and gated:
{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,25,32,64,128}, avx512,
forward-only, installed at codelets/n1_oop/ with a regen script. Coverage now
strictly exceeds FFTW's leaf set: everything FFTW ships (scalar 2-16, 20, 25,
32, 64 plus SIMD 128) plus our 17 and 19, with 9, 14, 15 newly emitted beyond
the previous production registry. Every radix passed the forward gate against
FFTW at 0 to 1.1e-14 relative (gate_n1_oop_swap.c, lane-blocked split layout,
direct-leaf strides V,1,V,1 count V).

The backward-swap identity is validated and becomes shipping policy. Calling
the forward codelet with both pointer pairs swapped, inputs (im,re) and
outputs (im,re), produces the unnormalized inverse against FFTW BACKWARD at
machine precision for all 22 leaves. It also holds through the full two-stage
engine (n1_oop + t1p with the unchanged forward twiddle table) at R=13 (N=169)
and R=32 (N=1024): engine bwd-swap at 7.1e-15 and 7.5e-15. Consequences:

* The _bwd half of the OOP codelet matrix is unnecessary. Backward transforms
  cost zero extra codelets, zero extra twiddle tables, one pointer swap at the
  call site. This is FFTW's own EXTRACT_REIM mechanism, which their interleaved
  SIMD cannot use but our split layout applies universally.
* General-N OOP now needs only: the n1_oop leaf set (done), executor wiring
  for a dst-writing first stage feeding the existing in-place stages, and an
  avx2 leaf set for the AVX2-only targets (future axis).

## 9. The AVX2 OOP leaf set and the ISA head-to-head

Full AVX2 leaf set generated and gated: all 22 radixes emit with --isa avx2,
compile under -mavx2 -mfma with zero zmm/k-register instructions
(objdump-verified), and pass both the forward gate and the bwd-swap gate at
0 to 1.5e-14. Installed at codelets/n1_oop_avx2/ with a regen script.

ISA race (bench_leaf_isa_race.c: avx512 leaf vs avx2 leaf vs FFTW guru, same
binary, same lane-blocked split data, round-robin, two runs per radix,
N*K ~= 256K elements per cell):

| radix | avx2/avx512 | fftw/avx512 | fftw/our-avx2 | FFTW's pick |
|---|---|---|---|---|
| 4 | 1.01-1.05 | 0.97-1.04 | 0.96-0.99 | n1fv_4_avx2_128 |
| 8 | 1.08-1.09 | 0.93-0.95 | 0.87 | n2fv_8_avx2_128 |
| 13 | 1.43-1.49 | 0.84 | 0.56-0.59 | n1fv_13_avx2 |
| 16 | 1.18-1.25 | 0.90-0.91 | 0.72-0.78 | n1fv_16_avx2 |
| 32 | 1.27-1.29 | 0.89-0.95 | 0.69-0.75 | n2fv_32_avx2_128 |
| 64 | 1.42 | 0.98-1.03 | 0.69-0.72 | n2fv_64_avx2_128 |

Three findings:

* The "avx2 might win cells on AVX-512 hardware" hypothesis is refuted for our
  architecture. avx512 wins or ties at every radix, with the gap growing from
  parity at R=4 to 1.42x at R=64. FFTW's avx2_128 planner preference does not
  transfer because the parallelism sources differ: FFTW vectorizes within one
  transform and pays width-scaling shuffle costs that make narrow vectors
  competitive for them; our lane-batched layout maps 8 transforms onto a zmm
  with zero shuffles, so full width is free. Consistent with the in-place
  finding (vfft-avx2 1.36x slower than vfft-avx512, geomean).
* Honest flag, REVISED by the column-layout re-race (bench_leaf_col_isa_race.c,
  one codelet call with count=K, matching FFTW's one-call internal batch): the
  earlier mid-radix avx512 leaf deficit was a harness artifact. Two opposing
  shape effects were uncovered. (1) The lane-blocked harness called codelets in
  8-lane crumbs (count=8, thousands of calls); fixing that, avx512 leaves BEAT
  FFTW at R=4 (1.22-1.28x), R=8 (1.32-1.42x), R=13 (1.27-1.50x), R=16
  (1.45-1.50x). (2) Column layout with power-of-two K puts elements K*8 bytes
  apart; at R=32/64 that is 64 streams on a 32KB-aliased stride, the classic
  L1/L2 set-conflict catastrophe, and our absolute cycles blow up 1.7-3x
  (avx512 R=64: 1.13M lane-blocked vs 3.3-3.5M column) while FFTW's contiguous
  interleaved transforms are immune. Correct per-radix shape: long-count
  one-call for small R, lane-blocked (element stride = 64B line) for large R,
  which is exactly what the production K-tiling machinery exists to do. With
  the right shape per radix, avx512 OOP leaves are parity (R=32/64,
  lane-blocked) to 1.5x ahead (R<=16, column one-call) of FFTW everywhere.
* The avx2-specific gap that survives the shape correction: best-shape avx2
  leaves vs FFTW's avx2-genus picks are 1.18-1.25x ahead at R=4, then behind
  at R=8 (0.79-0.80), R=13 (0.81-0.96), R=16 (0.85-0.90), R=32/64 (0.69-0.75,
  lane-blocked). A 10-30 percent avx2 leaf deficit at R>=8 is the real
  remaining signal: FFTW's codelets live on the same 16 ymm and do better
  here. Needs i9 validation (this host runs 256-bit on a server core) and is
  the avx2 emission-tuning target. avx2/avx512 ratio in best shape: 1.03
  (R=4) to 1.78 (R=8 column).
* Round-robin pollution caveat: in the column binary at R=32/64 our thrashing
  access pattern degrades cache state for the FFTW turn that follows, so the
  big-radix column fftw cycles are inflated; the lane-blocked binary is the
  cleaner FFTW reference at large R.
* FFTW picked an avx2-genus leaf at every size in this sweep, never avx512,
  even at R=64; its planner's measured preference on this host is consistent
  and real, just not applicable to our codelet architecture.

## 10. Files

* benchmarks/bench_t1p_rxr_vs_fftw.c           RADIX-parameterized two-stage vs guru (plan-print + gate)
* benchmarks/bench_n1_direct64_vs_fftw.c       direct-leaf N=64 race
* benchmarks/bench_r64_variants_interleaved.c  r64 leaf variant A/B/C, single binary
* benchmarks/bench_rxr_variants_interleaved.c  r7/r13 variant A/B/C/log3/specL3, single binary
* benchmarks/bench_r32_spec_interleaved.c      r32 generic-vs-spec x flat-vs-log3, single binary
* codelets for new radixes generated to /tmp/t1p_new (regenerable from section 2)

## 11. Status at pack time and continuation list

Shipped in this tree: full n1_oop leaf sets avx512 + avx2 (22 radixes each,
fwd-only, all gated incl. bwd-swap), t1p extension radixes 7/8/13 with the
spec winners for 7/13 (codelets/t1p_ext/), four single-binary race harnesses,
the per-radix gate, and this doc. All conclusions correctness-gated; all
ratios from round-robin single-binary timing.

Next, in value order:
1. MKL race for the OOP engines (specL3 vs split DFTI NOT_INPLACE, round-robin,
   N=1024/49/169): the headline product number, everything needed is in-container.
2. General-N OOP executor wiring: n1_oop dst-writing first stage + existing
   in-place t1/t1s stages; per-radix shape rule from section 9 (long-count
   one-call small R, lane-blocked large R) belongs in the planner.
3. avx2 emission tuning for leaves R>=8 (10-30 percent behind FFTW's avx2
   codelets on this host); validate on i9 first.
4. t1p avx2 generation for engine parity on AVX2-only targets.
5. Hygiene: K%8 guard on AVX-512 codelets; extend swap gate to log3/spec t1p
   variants; correction note in mkl_vs_vectorfft_1024_conclusion.md (its FFTW
   rows overstate the deficit ~1.8x); optional bench rename _interleaved ->
   _samebinary.
6. Off-container: EPYC/i9 confirmation of every ratio here; U=3 vs U=2 sweep.

## 12. The OOP strategy decision (design section for the general-N executor)

Decision: shallow Bailey/four-step with the fattest possible two stages,
recursing only when a factor exceeds cache or codelet size. Lane-blocked
intermediates, transpose always fused into the first stage's stores, twiddle
plus second DFT as one t1p call. This is not a proposal, it is the strategy
that already won; the alternatives were each killed by our own measurements:

* Stockham autosort: built and validated 2026-05-29, abandoned. log_R(N) full
  passes over the array; the 2026-05-30 layout-bottleneck finding (multi-pass
  element-major execution is memory-bound while codelets sit at their FP
  floor) is its indictment on CPU. Right answer on GPUs, wrong here.
* Deep factorization: engine_natural_oop_4stage.c (4x4x4x16) is the confirming
  negative; it fragments into many small strided calls and loses despite
  better numerical precision.
* FFTW-style recursion (engine_natural_oop.c, cache-resident intermediate) and
  shallow Bailey converge: the recursive step IS four-step applied recursively.
  The one-call 32x32 (engine_natural_oop_onecall.c) is its flattest, fastest
  expression and the validated winner vs MKL and FFTW.

Planner rules for the general-N wiring, all measured this session:

1. N <= largest leaf (<=128): direct n1_oop leaf, no twiddle stage (the
   two-stage at N=64 pays 15-20 percent for its transposed intermediate).
2. N = N1 x N2 with both factors codelet-sized: one-call two-stage, N2 drawn
   from the t1p set {4,7,8,13,16,32,64}; spec/specL3 codelets for small and
   mid radixes, generic for the r64 leaf.
3. Larger N: recurse, keeping factorization as shallow as possible, each
   sub-problem cache-resident.
4. Stage call shape per radix: long-count one-call for small R; lane-blocked
   (element stride = one cache line) for large R, never power-of-two-K column
   strides (the 32KB-alias catastrophe, section 9).
5. Backward: pointer swap at the API layer, zero bwd codelets (section 8).

What strategy work will and will not buy: it is the entire lever for general-N
coverage, for single transforms beyond cache, and for small-N plan choice. It
will not move the residual 5-7 percent cache-resident gap to MKL at N=1024,
which the disassembly study pinned to instruction scheduling.

## 13. Strategy isolation at N=1024: same codelets, four dataflows

The experiment the strategy decision deserved: identical radix-32 codelets
(generic n1_oop + t1p log3, same Q table, same arithmetic, same total
permutation), wrapped four ways (bench_strategy_iso_1024.c), round-robin in
one binary, all variants verified at 8e-15. The lane-blocked layout keeps
every variant on full-width contiguous vector ops (lanes ride inside the zmm,
so permutation absorption never costs gathers); this is a pure locality and
pass-count contest.

| variant | dataflow | K=128 vs A | K=512 vs A |
|---|---|---|---|
| A | fused Bailey: scatter stores, t1p in-place on dst | 1.000 | 1.000 |
| F | ping-pong Bailey: scatter into work, t1p work->dst | 1.08 / 1.10 | 1.09 / 1.16 |
| B | Stockham 2-stage: natural store, transpose in t1p loads | 1.16 / 1.16 | 1.20 / 1.21 |
| E | six-step: natural store, explicit transpose pass, t1p in-place | 1.40 / 1.40 | 1.37 / 1.42 |

Identical ordering in every run and regime: A < F < B < E. Decomposed:
in-place second stage beats ping-pong by 8-16 percent (hot dst, one fewer
buffer in the working set); store-side permutation absorption beats load-side
by 4-10 percent (stores drain in the background, strided loads sit on the DFT
dependency chain); one extra full pass costs 25-40 percent even cache-resident.
Choosing the wrong dataflow at N=1024 costs up to 1.4x with identical codelets,
which is the experimental justification for fixing the strategy by rule and
searching only parameters.

Side notes: FFTW drew its good plan at K=128 (fftw/A 0.87-0.89 against these
GENERIC codelets; the specL3 engine at 733K cycles still beats that draw,
consistent with section 7) and its bad plan again at K=512 (fftw/A 1.31-1.36).
The generic-vs-spec gap reproduced once more (A generic 880K vs specL3 733K at
K=128, ~17 percent).

## 14. Strategy sweep across N: the ordering is universal (this machine)

bench_strategy_iso_rxr.c generalizes section 13 to N = R1 x R2 (n1_oop(R2)
leaf stage, t1p_log3(R1) twiddle stage, dynamic twiddle row stride verified in
the generated source). Eleven cells, N = 49 to 8192, all gates <= 1e-14, K scaled to ~1MB per buffer.
Numbers are relative speed (A = 1.000, higher is faster):

| cell | N | A fused Bailey | F ping-pong | B Stockham | E six-step |
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

Findings:
* A < F < B < E at every cell, no exceptions, N spanning two orders of
  magnitude. The fused-Bailey rule survives its size-axis audit on this
  machine; the strategy axis stays a rule, not a search.
* A's margin is largest at small N (F/A 1.28-1.30 at N<=64) and smallest at
  4096 (1.035): when per-block arithmetic is tiny, the third buffer and lost
  dst-hotness weigh proportionally more.
* Factor order is a genuine searched parameter, not a rule: at N=512 the
  32x16 split beats 16x32 by 6 percent, while at N=2048 the 32x64 split beats
  64x32 by 8 percent. No single big-leaf or big-twiddle rule exists; this is
  exactly the per-cell residue the tuner ranks.
* The extra pass (E) costs 30-52 percent universally.
* Caveat: one host (Cascade Lake guest), one K band per cell plus regime spot
  checks; per-uarch audits remain the falsification mechanism.

Memory-regime spot checks at 4x K confirm and amplify (relative speed,
A = 1.000, higher is faster): 8x8 K=8192 F 0.707, B 0.681, E 0.621; 32x32
K=512 F 0.873, B 0.737, E 0.640; 64x64 K=128 F 0.843, B 0.689, E 0.672. The
wrong strategy bills at DRAM prices: extra buffers and extra passes cost more,
not less, as the working set leaves cache. Ordering unchanged.

## 15. Big multi-stage cell: fused Bailey vs six-step at N=262144

One-cell test of the conjectured six-step advantage at large N
(bench_strategy_bigN.c): N = 64x64x64 = 262144, K=8, ~100MB working set, well
past L3. A3 = fully fused recursive Bailey, three one-call codelet passes,
zero copies, but ALL passes carry 64-stream loads at 256KB stride. E6 =
six-step shape, two blocked transposes plus three contiguous codelet passes.
Both gates pass at 1.5e-14.

Result (relative speed, A3 = 1.000): E6 = 0.786 / 0.769 across two runs.
Fused Bailey wins by 1.27-1.30x even here. The pre-registered prediction was
the opposite (E6 to win on load-aliasing grounds) and it failed: the
256KB-strided loads are absorbed by L3 (slice hashing defeats set aliasing)
and hidden behind ~950 FP instructions per 64-load group, while E6's
transposes are pure traffic with nothing to hide behind. The section 13/14
lesson scales up unchanged: extra passes bill at DRAM prices, and that bill
exceeds the strided-load penalty at this size on this host.

Structural finding from the derivation, independent of timing: with affine
strides, a fused three-stage cannot produce natural order (digit-permutation
parity), so A3 outputs k1<->k2-swapped order within k_m while E6's transposes
deliver natural for free. The classic self-sorting trade. A natural-order A3
needs one cache-local fixup pass (64x64 swaps within 32KB chunks), far cheaper
than a full strided pass, so the ranking is unlikely to flip, but unmeasured.

Scope honestly stated: one host, one cell, naive (untuned) transposes in E6,
and "very big" in Bailey's original sense meant external memory; the six-step
domain that demonstrably remains is distributed memory, where FFTW also
hand-writes it (MPI layer). On this machine the fused rule extends past 100MB
working sets; the crossover, if it exists, is beyond what this container can
hold. EPYC audit item.

## 16. The OOP wiring plan (settled: fused four-step primary)

Everything lands in core/, nothing parallel.

1. Mode B general-N OOP: a branch at the existing stage-0 dispatch
   (stride_executor.h:1318, the same hook the C2R fused unpack uses). Stage 0
   calls an n1_oop leaf reading src and writing dst; stages 1..nf-1 run
   unchanged in-place on dst. stride_plan_t gains an oop flag; execute gains
   dst-taking entries. Inherits wisdom plans verbatim, K-tiling, calibration.
   Scrambled order. Constraint inherited from the same physics as FFTW's
   NO_DESTROY_INPUT: OOP implies DIT-oriented plans (DIF would destroy input).
2. Fused Bailey natural-order: a second plan kind in stride_plan_t carrying
   (R1, R2, Q table, variants); its execute function is the gated engine body
   from sections 13-15, not the stage loop. Planner ranks the two kinds per
   cell. Direct leaf for N <= 128 is a degenerate case of this kind.
3. API stays the proto API: plan_create_ex gains kind/oop, execute gains
   (sr,si,dr,di) variants, bwd is a pointer-swap wrapper, K%8 guard in
   plan-create validation, rule predicates (leaf<=128 direct, aliasing mask,
   specL3/generic variant defaults) where factors are validated today.
4. Tuning lands in the owning files: OOP CPE rows (shape-keyed) into
   estimate_plan.h tables, candidates through the existing screened/patient
   machinery, wisdom entries gain the node-kind field. OOP leaf registry:
   hand-written table for the 22 leaves first, generator-emitted later.

Sequencing: Mode B end-to-end one session; Bailey kind plus predicates a
second; tuner third; then validation sweep and the MKL race; hardware audits
off-container. Out of scope: Stockham/deep-CT (audit harnesses only),
six-step (distributed-memory domain), Tier-B pin, primes beyond 19.

## 17. OOP engine landed in core/ (sections 16 items 1-3 complete)

Files: core/oop_execute.h (Mode B), core/oop_plan.h (plan kind + rules),
core/oop_leaf_registry.h (hand-written 11-arg registry: 22 leaves + 7 t1p),
benchmarks/test_oop_execute.c, benchmarks/test_oop_plan.c,
benchmarks/bench_bailey_col.c (column-Bailey derivation gate).

Mode B (vfft_proto_execute_fwd_oop): stage 0 via the n1 codelets'
contractual OOP signature, stages 1.. in-place on dst through a shifted
shallow plan view into the generic loop. Gates: fwd BIT-IDENTICAL to the
in-place generic dataflow, src preserved, bwd-swap bit-identical, at
1024[8,8,16], 4096[16,16,16], 1024[32,32], 2310[2,3,5,7,11], 2000[20,10,10],
169[13,13]. Workflow timing (speed vs memcpy+in-place = 1.000): OOP direct
0.998-1.162; pure in-place (destroys input) 1.367-1.592, which is the
working-set price of input preservation itself, not implementation cost.

Column-layout Bailey (the layout-unified plan kind): re-derived for the
proto/MKL split convention (element e at e*K + t). s1 = R1 long-count
n1_oop(R2) calls, fused-transpose stores; s2 = one t1p_log3(R1) call
in-place, K-replicated twiddle table (grp_tw memory model); natural order
X[k2 + R2*k1]. Gates 7e-15 to 1e-14 at 13x13, 32x32, 32x64, 16x32. Races:
1.87x vs FFTW at 169; LOSES 0.63-0.68x at pow2 NxK cells because the s2
stride R2*K lands on the 32-64KB set period with 16-32 streams, exactly
rule 4. Consequence baked into the planner: the aliasing mask checks both
stages (j-stride multiple of 512 doubles with more than 8 streams) and
masked cells fall to Mode B, whose wisdom factorizations use radixes that
fit associativity.

vfft_oop_plan_create rules, all gated live in test_oop_plan.c:
K%8 rejected outright (kills the heap-corruption mode); N<=128 -> direct
LEAF; unmasked divisor pair -> BAILEY2 (fattest leaf, then balance; tuner
ranks pairs later); else MODEB from supplied factors (wisdom integration
is phase 4). Demonstrated K-dependence: N=1024 K=128 -> MODEB, N=1024
K=120 -> BAILEY2 passing vs FFTW at 8.2e-15. Backward for every kind is
the pointer swap; LEAF and BAILEY2 bwd gate against FFTW BACKWARD at
machine precision.

v1 scope: single-threaded, avx512 registry, tier-1 plan-shaped executors
not OOP-wired (documented 5-6 percent), Mode B requires factor lists until
the wisdom/tuner phase, mb plans share the proto path's Phase-1 no-destroy
ownership.

## 18. Phases 4-5: auto-planner, tuner, sweep, and the MKL race

Phase 4 (core/oop_auto.h): vfft_oop_plan_create_auto resolves with no
caller factors: tuned-pair hints, then the rule spine, then wisdom-backed
MODEB (built DIT regardless of the entry's DIF preference; fidelity caveat
noted). vfft_oop_tune_pairs is the entire searched residue: same-binary
round-robin over unmasked pairs PLUS the direct leaf at N<=128 (a hint must
never shadow a faster leaf; at 64/K=512 the tuner measured the 16x4 pair
~2 percent ahead of the leaf and kept it, measured over assumed). Pair
spread measured up to 24 percent (512/K=120, five candidates); the static
preference is balanced-first, tuner-corroborated, tuner overrides per cell.

Aliasing mask refined to the MEASURED boundary: the catastrophe requires
the 32KB period (stride multiple of 4096 doubles) with streams beyond
8-way associativity; 4KB-only aliasing is absorbed by L2 under the DFT
arithmetic (169/K=512, stride 52KB, runs BAILEY2 at 2.7-2.8x vs MKL).
Refinement upgraded 6 sweep cells from MODEB to natural-order BAILEY2 and
made 1024/K=256 both natural AND faster (3.22M vs 3.57M cyc).

Phase 5a validation sweep (test_oop_sweep.c) over the production wisdom
table, stride 2, cap 2M elements: 54 cells tested, 5 LEAF + 11 BAILEY2 +
38 MODEB, ZERO failures. 31 cells skipped below the K%8 lane contract,
which is real for every kind on this path: the proto avx512 7-arg codelets
are 8-lane granular and K=4 on exact-size buffers overruns leg slices
(found as heap corruption, now rejected at create).

Phase 5b THE MKL RACE (bench_oop_vs_mkl.c): OOP engine vs MKL DFTI,
both out-of-place, split storage, column layout, single thread, same
binary, round-robin min-of-rounds. Higher is faster:

| cell | kind | gate | order | speed vs MKL |
|---|---|---|---|---|
| 64 K=512 | BAILEY2 16x4 (tuned) | 3e-15 | natural | 3.68 |
| 128 K=512 | BAILEY2 4x32 (tuned) | 5e-15 | natural | 3.24 |
| 169 K=512 | BAILEY2 13x13 | 7e-15 | natural | 2.65 |
| 512 K=120 | BAILEY2 32x16 (tuned) | 7e-15 | natural | 2.56 |
| 1024 K=120 | BAILEY2 16x64 (tuned) | 9e-15 | natural | 2.37 |
| 1024 K=256 | BAILEY2 8x128 | 8e-15 | natural | 2.51 |
| 4096 K=256 | MODEB | bit-exact | scrambled | 2.73 |
| 2310 K=32 | MODEB | bit-exact | scrambled | 3.51 |

Geomean ~2.9x. Earlier run (pre-refinement) drew 2.07-3.58 on overlapping
cells; ratios stable within the usual bounds.

Honesty block: (1) single noisy Cascade Lake guest, EPYC/i9 confirmation
pending as always. (2) MKL's REAL_REAL split storage is its weaker
configuration; the May-31 interleaved OOP comparison at N=1024 measured
1.07-1.44x, so split-vs-split overstates the margin a user on interleaved
data would see. The split column layout IS our native and HFT-relevant
format, but both numbers belong in any public claim. (3) MODEB rows
deliver scrambled order; MKL delivers natural; a natural-order reorder
pass on MODEB cells is not measured. (4) MODEB cells run the generic loop
(tier-1 not OOP-wired, documented 5-6 percent on the table).

Remaining: phase 6 hardware audits; hygiene items from section 11.

## 19. AVX2 wiring complete

The OOP engine now builds per-ISA (same model as the proto executor):
oop_leaf_registry.h selects avx512 or avx2 symbol sets from __AVX512F__
(override VFFT_OOP_FORCE_AVX2) and exports VFFT_OOP_GROUPW (8 / 4), which
parameterizes the K-replicated twiddle tables (avx2 codelets step b += 4
and index tw[row*(me/4) + b/4], verified in generated source). avx2 t1p
codelets (plain + log3, 7 radixes) generated, pure, installed at
codelets/t1p_avx2/.

Gates: avx2 builds of test_oop_plan and test_oop_auto ALL PASS (LEAF,
BAILEY2 with K/4 replication, pow2 MODEB bit-exact, bwd everywhere,
tuner ranking live: 16x32 and 32x32 winners again). avx512 regressions
re-run green after the shared-header changes.

Structural finding: the lane-blocked (V=8) engine layout is
avx512-width-native. With 4-wide groups the codelet's affine b*OG store
model cannot express the V=8 scatter (needed address 256*(p>>3) + 8j +
(p&7), non-affine in p), so the lane-blocked engine swap gate fails on
avx2 BY CONSTRUCTION, not by bug; leaves pass, and the shipped
column-layout kinds are group-width agnostic. An avx2 lane-blocked engine
would use V=4 blocking (not built; no product need).

Known limitation: odd-N Mode B is unavailable on avx2 because the avx2
in-place codelet set is pow2-only; the registry's odd avx2 symbols are
satisfied by abort-stubs (benchmarks/avx2_inplace_stubs.c) so builds link
and any accidental call fails loudly. Odd N on avx2 is served by the OOP
LEAF/BAILEY2 kinds (full 22-leaf avx2 set). Generating the full avx2
in-place set is the proper fix, queued with phase 6.

avx2 vs MKL preview on THIS host (our code 256-bit, MKL free to dispatch
avx512; the honest lower bound for the i9 where MKL is also 256-bit):

| cell | kind | speed vs MKL |
|---|---|---|
| 64 K=512 | BAILEY2 4x16 | 2.02 |
| 128 K=512 | BAILEY2 4x32 | 1.90 |
| 169 K=512 | BAILEY2 13x13 | 1.78 |
| 512 K=120 | BAILEY2 16x32 | 1.81 |
| 1024 K=120 | BAILEY2 32x32 | 1.67 |
| 1024 K=256 | BAILEY2 8x128 | 1.20 |
| 4096 K=256 | MODEB | 1.58 |

All gates at machine precision; i9 validation remains the deciding
measurement for avx2 claims.

## 20. The avx2 vs FFTW question: gap NOT closed

Two races answer it. First, our avx2 build vs the avx512-enabled FFTW
(its planner mixed t2fv_8/t1fv_64/t1fv_32_avx512 twiddle stages with avx2
leaves): we lose 0.40 to 0.76 with MODEB 4096 at 1.06. That race is
asymmetric, so an avx2-only FFTW 3.3.10 was built (1175 avx2 symbols,
avx512 disabled at configure, the true i9 proxy) and the race rerun
SYMMETRIC, both sides 256-bit, same binary:

| cell | kind | gate | speed vs FFTW-avx2 |
|---|---|---|---|
| 64 K=512 | LEAF | 6e-15 | 0.52 |
| 128 K=512 | BAILEY2 4x32 | 5e-15 | 0.72 |
| 169 K=512 | BAILEY2 13x13 | 6e-15 | 0.76 |
| 512 K=120 | BAILEY2 32x16 | 6e-15 | 0.89 |
| 1024 K=120 | BAILEY2 32x32 | 8e-15 | 0.67 |
| 1024 K=256 | BAILEY2 8x128 | 7e-15 | 0.50 |
| 4096 K=256 | MODEB | bit-exact | 1.09 |

FFTW-avx2 wins six of seven cells by 1.12 to 2.0x; MODEB takes the
largest cell. The N=64 row is the clean diagnostic: one codelet against
one codelet (our n1_oop 64 vs FFTW n1fv_64_avx2), 0.52, so FFTW's avx2
codelet body is roughly 2x our avx2 emission at R=64. Consistent with
the section-9 bare-leaf gap (10 to 30 percent at R >= 8) compounding and
growing with leaf size. The strategy is not the problem: the identical
strategy wins on avx512; the deficit tracks codelet bodies. The avx512
emission went through the full sections 1-8 tuning campaign; the avx2
emission never did, and 16 ymm registers against avx512-shaped
scheduling is the obvious first suspect (FFTW's avx2 genus is scheduled
for 16 registers; its winning plans also prefer 8/16-point leaves).

Consequences: on the i9, MKL remains beatable (we win 1.2 to 2.0x here
even with MKL free to dispatch avx512; on the i9 MKL is also 256-bit),
but FFTW-avx2 wins these cells as things stand. Work item: avx2 emission
tuning in the generator (16-register scheduling, U sweep for avx2,
possibly an avx2-specific radix preference). Caveats: 256-bit code on a
server core is not an i9; tuner pair picks are noisy between runs on
this host (16x32 vs 64x8 observed at 512/K=120); the i9 audit stays the
deciding measurement.

Harness note for the record: the first bench_oop_vs_fftw gates printed
exactly 0e+00 and were VOID. FFTW PATIENT planning scribbles on its
arrays during measurement; the input was filled before planning, so the
gate compared against FFTW-of-garbage and the degenerate mm==0 branch
passed silently. Fixed (refill after planning, degenerate fails), rerun,
real gates 5e-15 to 8e-15, timings unchanged. bench_bailey_col did not
share the bug (fills after planning).

## 21. Hygiene closeout (the section-11 + section-19 tail)

1. Full avx2 in-place codelet set generated: 216 codelets (12 radixes x
   9 families x fwd/bwd) via the PRODUCTION recipe in
   generator/scripts/generate_codelets.sh, --in-place rio convention
   (the first attempt used bare flags, produced the contiguous-layout
   family, and segfaulted at the 2310 cell; the flag guide is
   generator/scripts/README.md and the script is authoritative).
   Installed at codelets/inplace_avx2/, compiled with the production
   avx2 config (-mavx2 -mfma -march=haswell), all pure, zero missing
   symbols vs the registry. Odd-N Mode B is LIVE on avx2: 2310 BITEXACT
   fwd + bwd + preserve in the avx2 plan gates, avx512 regressions
   green, and the avx2-vs-MKL race gains its missing row: 2310 MODEB at
   2.48x (MKL free to dispatch avx512). The abort-stubs are retired
   (benchmarks/avx2_inplace_stubs.c kept, marked RETIRED).

2. Correction note appended to mkl_vs_vectorfft_1024_conclusion.md: its
   FFTW rows were cross-binary and overstate ~1.8x; same-binary races in
   this document supersede them; the MKL rows stand.

3. Swap-gate extension measured (gate_n1_oop_swap.c gains T1P_LOG3 and
   SPEC modes): log3 t1p engine fwd + bwd-swap at R=13/32/64 pass at
   7.1e-15 to 1.3e-14; stride-specialized R=32 engine (both specL3 and
   plain spec, baked strides, 7-arg ABI) passes fwd + bwd-swap at
   machine precision. The pointer-swap identity now has measured cover
   on every codelet family in the tree.

Lead filed for the avx2 leaf-performance revisit: per the generator
README, GH (Goodman-Hsu pressure-aware scheduling) auto-fires for
IN-PLACE avx2 codelets at R >= 32; whether the --oop emission path
engages GH at all is unverified and is the first thing to check.

## 22. Diagnosis: the OOP emitter discards the scheduler (avx2 gap mechanism)

The question that cracked it: do the n1 in-place and the OOP leaf, both
first-stage no-twiddle DFTs, get the same generation flags? They do not,
and structurally cannot. The dispatch in gen_radix.ml passes
~scheduler ~gh ~spill to Emit_c.emit_codelet on the in-place branch, but
the --oop branch calls Codelet_oop.emit_codelet with structural config
only; codelet_oop.ml line 769 states it outright: "no SU scheduler
(topological order within each pass)". The auto-rules that fire above
the dispatch (SU universal default; GH on avx2 at n >= 32, documented
+4-8 percent; SU on n1 avx2 R=8 documented 1.35x) are computed and then
silently discarded for the OOP family.

Artifact evidence (spill movs, rsp traffic, avx2):

| codelet | spill movs |
|---|---|
| in-place n1_32 (SU) | 289 |
| OOP n1_32 (no SU) | 548 |
| in-place n1_64 (SU) | 817 |
| OOP n1_64 (no SU) | 881 |
| in-place t1_dit_64 (SU+GH+recipe) | 882 |
| OOP t1p_64 log3 (no SU) | 1131 |

Calibration: SU absence explains 1.9x spill traffic at R=32 and 28
percent at t1p_64, but NOT the whole 2x runtime gap at the 64-leaf,
where even SU in-place spills 817 times. R=64 avx2 spills heavily under
any scheduler, which is exactly why the in-place cost model fires
should_block_n1 (blocked construction) there, and why FFTW's avx2
planner prefers 8/16-point leaves. The OOP path is missing BOTH
responses to 16-register pressure: the scheduler and the structural
blocking, and our avx2 plan preferences still favor fat leaves.

Fix list for the leaf-performance work, in order: (1) wire SU and GH
into Codelet_oop emission; (2) blocked construction or an avx2-specific
small-leaf preference for the OOP kinds (the tuner already measures;
the candidate set needs the small-leaf bias); (3) re-race vs FFTW-avx2.
avx512 is unaffected (32 registers; the gap never existed there).

## 23. The full oversight inventory: the OOP emitter skips the optimizer
### CORRECTION (same day, before wiring began)

The pass table below is WRONG about the algsimp cascade. codelet_oop.ml
delegates to lib/pipeline.ml (Pipeline.prepare_codelet), the shared
cascade module, with aggressive/reassoc/fma env gating intact; my grep
for direct Algsimp.* calls missed the indirection. What survives of the
diagnosis: the SCHEDULER gap is real (codelet_oop orders nodes by
topological sort-by-tag; no SU parameter, no GH; its own line-769
comment confirms), and pipeline-vs-gen_radix drift needs an audit since
gen_radix.ml still carries an inline copy of the cascade. The spill
deltas (548 vs 289 at n1_32 avx2) are therefore attributable to
scheduling, not missing DAG passes, sharpening rather than weakening
the SU/GH wiring case. Original (partly wrong) section kept below for
the record.


Confirmed as a design oversight, and the scope is now exact. The OOP
family (codelet_oop.ml) rebuilds its DAG with the SAME blocked
constructors as production (dft_expand_n1_blocked under should_block_n1,
dft_expand_twiddled_spill under should_spill, the doc-58 machinery that
delivered the 47-58 percent n1 runtime wins), then runs Algsimp.
of_assignments and STOPS. The in-place ladder it skips, in order
(gen_radix.ml 290-400):

| pass | in-place | OOP |
|---|---|---|
| of_assignments (~reassoc per needs_reassoc) | yes | yes (reassoc default) |
| dedup_sub_pairs (x2, pre and post factor) | yes | NO |
| factor_common_muls (~aggressive, primes) | yes | NO |
| factor_by_atom (~aggressive, primes) | yes | NO |
| collect_m + deep_collect fixed point (M-project) | yes | NO |
| share_subsums (composites/pow2) | yes | NO |
| transposition loop (untwiddled only) | yes | NO |
| fma_lift (single_use policy, doc 57) | yes | NO |
| SU scheduler (Sethi-Ullman) at emit | yes | NO (topological per pass) |
| GH pressure mode (avx2, n >= 32) | yes | NO (parameter absent) |

So blocking survived the port; the entire optimizer and scheduler did
not. This jointly explains the ISA asymmetry: on avx512 (32 regs) the
missing passes cost op-count only and the OOP leaves stayed near FFTW
parity; on avx2 (16 regs) the missing pressure machinery (SU, GH,
fma_lift's register-allocator interactions) is decisive, hence the
0.5-0.9 plan-level losses to FFTW-avx2 and the 548-vs-289 spill counts.

Fix shape for the leaf-performance session: factor the gen_radix.ml
pass ladder (of_assignments through fma_lift, with the aggressive/
is_direct gates) into a shared function consumed by both branches, and
thread scheduler+gh into codelet_oop's per-pass node ordering. Then
regenerate BOTH ISA OOP sets, re-run the complete gate suite (every
OOP codelet changes), and re-race vs FFTW-avx2 and MKL.

## 24. Wiring session outcome: the gap is not a missing wire

Scoreboard of the excavation, in order. Hypotheses KILLED by reading or
measurement: (1) missing algsimp cascade (wired via shared
lib/pipeline.ml, zero pass drift vs gen_radix verified); (2) missing
SU+GH scheduling (Tier C in codelet_oop implements cluster-local SU with
the GH auto-rule; the file's own line-769 comment is stale); (3) missing
value fences (wired via prep.fence_enabled with save/restore; my
redundant patch reverted same day); (4) missing register allocator
(Regalloc is gated to log3+avx512+R<=32 on the IN-PLACE path too; both
paths run unallocated for the codelets in question); (5) blocking
asymmetry (both paths block at n>=25; the PASS-grep was an artifact).

LANDED: per-ISA uarch in Tier C. codelet_oop hardcoded
sapphire_rapids_avx512, whose pressure_threshold=24 made GH effectively
inert on 16-register targets (raptor_lake_avx2 threshold is 12) and fed
SU wrong-ISA tables. Fixed to select by vec_regs. avx512 output
byte-identical; avx2 n1_32 spills 548 to 531; both OOP codelet sets
regenerated, all plan gates pass, FFTW-avx2 race unchanged within noise
(0.51-0.92).

The reframed conclusion: OOP emitter machinery is at near-parity with
the in-place emitter. The residual 531-vs-289 spill delta is largely
function shape (strided OOP edges carry more live values) plus emission
details, and closing it entirely would only reach the in-place
generator's own avx2 level, which section 9 already measured 10-30
percent behind FFTW's avx2 genus at bare leaves (2x at the 64-leaf).
The planner side already mitigates (the tuner picks small-leaf pairs,
4x16 at 64, 4x32 at 128, matching FFTW's preference). What remains is
an avx2 EMISSION QUALITY campaign for the generator itself, uarch table
calibration for 16-register targets, scheduling experiments, FFTW's
n1fv structure as the per-codelet benchmark, run exactly like the
avx512 campaign of sections 1-8 was run, with its own measurement loop.
That is a project, not a patch, and it benefits BOTH codelet families.

## 25. The decomposition: op count vs scheduling, three ways

Pre-registered bets: Tugbars bet op count, the session bet scheduling.
Op count FALSIFIED at the DAG level. genfft header counts (FMA form,
per VL=2) vs our compiled vector arith (per 4-transform group),
per-transform: R8 13 vs 13.0, R16 36 vs 36.0, R32 97 vs 93, R64 238 vs
228, R128 582 vs 541.5. Parity to within 4-8 percent; identical to the
digit at 8 and 16. The cascade is genfft-grade arithmetic.

Three-way compiled counts (arith/spill movs; ours per 4 transforms,
FFTW per 2; normalize accordingly):

| R | in-place avx2 | OOP avx2 | FFTW avx2 |
|---|---|---|---|
| 8 | 52/6 | 52/48 | 46/0 |
| 16 | 144/113 | 144/329 | 65/15 |
| 32 | 386/289 | 386/531 | 171/65 |
| 64 | 952/817 | 952/881 | 425/179 |
| 128 | 2328/2141 | 2328/2055 | 1019/527 |

Findings: (a) in-place and OOP arithmetic identical (shared Pipeline,
confirmed at the instruction level). (b) Per transform, in-place spills
2.0-2.3x FFTW at R=32/64/128 WITH the full recipe engaged, and at R=128
the OOP path is marginally BETTER than in-place (513.8 vs 535.3 per
transform). The spill problem is GENERATOR-WIDE on avx2, not
OOP-specific. (c) The monolithic gap is real and localized: R=8 and
R=16 (below blocking threshold) show OOP at 8x and 2.9x in-place spill
traffic, because the OOP monolithic path orders by tag with no SU
(Tier C covers only the spill path). Note FFTW's compiled arith per
transform runs slightly below ours at composite sizes (imperfect FMA
fusion on our side adds ~10 percent compiled arith over the DAG count).

Campaign structure that follows: Tier 1 (bounded): SU over the OOP
monolithic node set, prediction n1_16 329 to ~78-113, n1_8 48 to ~6;
directly hits the avx2 planner's chosen small-leaf pairs. Tier 2 (the
project): blocked-path scheduling for the GENERATOR, in-place included,
since even SU+blocking leaves 2.0-2.3x FFTW spill traffic per transform
at R>=32. Candidates: existing --bisect / --bb (offline codegen can
afford minutes of search), GH threshold sweep 8-16, per-ISA CT
factorization (8x8 at R=64 was chosen on avx512; 4x16 or 4x4x4 may suit
16 registers), per-ISA blocking threshold, or a genfft-style
recursive-bisection partitioner as a fourth scheduler. Shared machinery
means every win lands in both families.

## 26. Liveness autopsy at R=16: the case for the bisection scheduler

Method: replay each codelet's emitted statement order and count peak
simultaneously-live compute values (constants excluded; FFTW FMA
variant, lines 37-188 of n1fv_16.c). Pre-registered: FFTW ~12, ours
materially above 16. Result, stronger than predicted:

| emission order | peak live (R=16) | measured spill movs/transform |
|---|---|---|
| FFTW bisection | 8 | 7.5 |
| ours SU (in-place) | 35 | 28.3 |
| ours tag order (OOP) | 57 | 82.3 |

genfft's recursive bisection holds a 16-point FFT to 8 live values,
half the avx2 register file. SU peaks at 2.2x the budget, tag order at
3.6x. The peaks track the measured spill columns. Caveat recorded:
genfft inlines single-use values at source level while we name all
temps and rely on gcc; the compiled spill counts prove gcc does not
rescue our ordering, so the comparison stands.

Convergence statement: three independent measurements now give the
same verdict. (1) The MKL N=1024 assembly audit (our 32x32 log3 OOP
does less arithmetic than MKL, which stays close anyway). (2) The FFTW
three-way decomposition (section 25: arithmetic parity, 2-3x spill
traffic). (3) This liveness replay (peak pressure 4-7x FFTW's at equal
op count). Background theory agrees: SU is optimal for trees, FFT DAGs
are maximally shared, and Frigo's bisection is register-count-oblivious
with asymptotically optimal register traffic (Hong-Kung bound). The
remaining frontier is the schedule.

Plan of record: Tier 1 (an afternoon): SU over the OOP monolithic node
set, closes OOP to in-place at the planner's hot small leaves. Tier 2
(the campaign): implement genfft-style recursive bisection as a fourth
Emit_c.scheduler over the shared DAG layer, replacing rather than
extending the GH/threshold machinery; validate at R=16 against peak
live 8, then sweep the families. Existing untried levers (--bisect,
--bb with time budget) get audited first in case the machinery already
exists under those flags.

## 27. Bisection repair, part 1: components and subset-relativity

The --bisect audit found a full Frigo port in lib/schedule.ml
(RED/BLUE waves, Seq recursion, bisection_schedule public API) that had
never been raced. Raced, it lost badly: 245/1047/2997 spill movs at
R=16/32/64 avx2 (SU recipe: 113/289/817), with all loads hoisted to the
top of the emission. Two fidelity gaps vs genfft/schedule.ml (the
original is on disk and served as the spec):

1. genfft recurses through connected_components at every level before
   partitioning; our port bisected raw blobs. LANDED:
   connected_components_of + recursion in schedule_nodes.
2. genfft rebuilds the dag per sublist (makedag), making inputs,
   outputs, and both wave conditions SUBSET-RELATIVE. Our port used
   global preds/succs, so pure-compute subsets (no store sinks) had no
   outputs, the BLUE wave never seeded, cuts degenerated, and the
   topological fallback front-loaded all loads. LANDED: subset-relative
   bisect (member table, preds_in/succs_in).

Result cascade (in-place avx2 n1 spill movs):

| R | SU recipe | bisect v1 | +components | +subset-relative | FFTW |
|---|---|---|---|---|---|
| 16 | 113 | 245 | 176 | 152 | 15 |
| 32 | 289 | 1047 | 882 | 651 | 65 |
| 64 | 817 | 2997 | 2347 | 1665 | 179 |

Emission cadence at R=16 is now correct at the leaves: load, load,
sub, add, repeating, the genfft pattern. The remaining 2x-vs-SU and
10x-vs-FFTW lives in the middle layers. Next suspects from the genfft
source, in order: (a) annotate.ml's reorder pass, a greedy
overlap-maximizing ordering of Par siblings (our components concatenate
in node-id order, so cross-component consumers land far from
producers); (b) linearize's balanced Par splitting;
(c) schedule_for_pipeline appears gated off by default in genfft
(flag absent from the stamped n1fv command lines), so it is NOT the
explanation and is parked. Verification loop per stage stays: cadence
check, spill counts vs the table above, liveness replay, then races.

## 28. Bisection repair, part 2: reorder, constants, and two topology falsifications

LANDED (all gated behind --bisect; production recipes untouched,
su_schedule_subset unmodified): (1) reorder_components, the port of
genfft annotate.ml's greedy overlap-maximizing block ordering;
(2) constant decoupling in connected_components_of: in genfft's IR
constants are inline literals with no dag presence, while our
hash-consed NK_Const nodes welded every CT sub-DFT into one component,
silently disabling the decomposition. Specials consumed by one
component attach to it; shared specials hoist ahead of all components
(an earlier attach-to-min-id version was a use-before-def LEGALITY BUG,
caught by gcc at R=32 and fixed by structural hoisting exempt from
reorder).

Spill table, in-place avx2 n1 (arith unchanged unless noted):

| R | bisect v1 | final bisect | SU recipe | FFTW |
|---|---|---|---|---|
| 16 | 245 | 137 | 113 | 15 |
| 32 | 1047 | 421 | 289 | 65 |
| 64 | 2997 | 1119 | 817 | 179 |

Bisection improved 1.8-2.7x from the degenerate state but STILL LOSES
to SU at every size. The wire-as-avx2-default gate (beat 113/289/817)
is NOT met; no production wiring, no races.

FALSIFIED, with the floor analysis that reframes the campaign:
(a) Split-radix topology (VFFT_SPLIT_RADIX=1 + bisect): spills flat
(126/425/1076), peaks worse at 16/32, arith UP 8-20 percent (fma_lift
gated off for SR). Our SR construction materializes the same wide
seams. (b) Unbalanced CT(2,N/2) via VFFT_CT_FACTOR: worse than
defaults under both schedulers (su 119/343/1018); the streaming-pairs
idea requires a scheduler that interleaves sub-DFT cones
output-synchronized, which neither SU nor bisect attempts. (First
attempt used the wrong override syntax, 2x8 vs 2,8, and silently
measured the defaults; caught by identical arith counts.)

THE FLOOR: liveness peaks now explain the ceiling. Our CT(4,4) R=16
dag has a 32-real seam at the pass boundary; any schedule that
completes PASS 1 before PASS 2 holds all of it, and SU's peak of 35 is
already AT that floor. Scheduling is nearly exhausted on our dag
shapes. FFTW's peak of 8 complex values is a property of genfft's dag
CONSTRUCTION: split-radix where the even half STREAMS through the
combine layer as it is produced, never accumulating, leaving only the
small odd parts resident. Their scheduler then merely respects a
structure that is already narrow.

Frontier shift: the next stage is DAG CONSTRUCTION, not scheduling.
Build a streaming pow2 expansion (genfft-style split-radix where
combines consume the recursive half incrementally), then let the
repaired bisection schedule it. The repaired scheduler is necessary
but not sufficient; the topology sets the floor it schedules against.

## 29. Spill accounting correction: SIMD spills vs GPR traffic

Tugbars's catch: the spill counter (mov + rsp regex) conflated vector
spills (vmovapd ymm to/from stack, the schedule-pressure signal) with
GPR rsp movs (address arithmetic, scalars). Separated counts, avx2 n1,
format vec/gpr:

| R | in-place SU | bisect final | OOP | FFTW |
|---|---|---|---|---|
| 8 | 6/0 | - | 22/33 | 0/0 |
| 16 | 90/27 | 109/32 | 226/111 | 15/0 |
| 32 | 179/125 | 292/146 | 272/265 | 65/0 |
| 64 | 561/308 | 855/312 | 765/117 | 179/0 |

Corrections to the record: sections 25-28 used conflated counts. Pure
SIMD ratios vs FFTW are 6x / 2.8x / 3.1x at R=16/32/64 (slightly
kinder than stated). Every directional conclusion survives: bisect
still trails SU in vector spills, the seam-floor analysis is about
vector values and is unaffected, OOP small-R monolithic remains worst.

NEW FINDING: FFTW carries ZERO GPR stack traffic at every size; we
carry up to 308 GPR movs at in-place R=64, growing with radix. This is
a second, previously invisible cost lane: address arithmetic. Our
in-place indexing (k + c*ios for many distinct c) makes gcc
materialize and spill offset values on 16 GPRs; FFTW's stride macros
and pointer discipline keep addressing register-resident. Notably OOP
R=64 carries only 117 GPR movs vs in-place's 308 (strided-edge
addressing is cheaper), so the two families bleed in different lanes.
Work item, independent of the vector scheduling wall: addressing
emission (per-group base pointers / strength-reduced offset chains),
likely cheap, benefits the hot loop directly. Convention from here:
all spill numbers reported as vec/gpr pairs.

## 30. Discovery: "SU" is not SU. The scheduler stack inventoried

Tugbars's hypothesis, confirmed by reading lib/schedule.ml end to end:
since repaired-Frigo bisection cannot beat our "SU", our SU must not be
pure. It is not. The picker (su_schedule / su_schedule_subset) is an
eight-layer demand-driven list scheduler: (1) lazy loads, never fired
while arithmetic is ready, demand-only in source order; (2) sink-first,
empty-user nodes fire immediately (documented: R=17 t1_dif 176 to ~115
vmovapd, matching hand); (3) cp_dist descending, latency-weighted
critical path from uarch tables; (4) Sethi-Ullman numbering, THE ONLY
TEXTBOOK REMNANT, demoted to third tie-breaker; (5) stable tag order;
(6) Goodman-Hsu pressure mode, live-set tracking with remaining_users
and a uarch threshold switch; (7) port-class balancing, P0/P1 vs P5
Ice Lake dispatch modeling; (8) symbiosis with the recipe's structural
blocking on big codelets.

Layers 1-2 are greedy per-instruction approximations of exactly what
bisection achieves structurally (loads attach to consumers; ranges die
at earliest stores), independently evolved. The bisect-vs-SU race was
therefore never Frigo vs Sethi-Ullman: it was register-oblivious 1999
structure vs a months-tuned 2025 multi-criteria machine, both pressing
the same seam floor (peaks 35 vs 32 at R=16). Within 20 percent is a
respectable showing for structure alone, and the scheduling frontier
on current CT topologies is confirmed closed from both directions.

Standing requirement for the streaming-dag stage: re-race BOTH
schedulers on the new topology. The list scheduler's greedy policies
may or may not discover the streaming order; bisection's recursion is
the natural fit for a graph that is finally narrow by construction.
Naming note for future sessions: refer to the production scheduler as
THE LIST SCHEDULER; "SU" survives only in flag names.

## 31. Research: our eight layers vs genfft's shipped pipeline

Verified from genfft/magic.ml defaults plus the annotate.ml driver:
schedule_for_pipeline, reorder_insns, reorder_loads, reorder_stores all
default FALSE and are absent from the command lines stamped into the
shipped codelets (section 27's parking of schedule_for_pipeline was
correct, now verified rather than assumed). The ACTIVE genfft pipeline
is exactly: bisection + connected components (structure), overlap
reorder + balanced linearize, collect_buddy_stores (SIMD re/im store
pairing, an interleaved-format need our split format does not have),
liveness analysis, and -compact -variables 4 declaration scoping.

Comparison verdict: their entire scheduler contains ZERO numbers. No
latency tables, no pressure thresholds, no port models, no per-ISA
anything; the only constants are VL and the 4 of -variables. Every
property our list scheduler computes explicitly (lazy loads, sink-first
range ending, critical path, pressure mode, port balance) is emergent
from graph shape in theirs. On equal topology our explicit machine WINS
(sections 27-28: repaired bisection trails the list scheduler by ~20
percent on our dags). FFTW's real-race advantage is therefore entirely
below the scheduling layer: fftgen's dag construction produces graphs
narrow enough that a numbers-free scheduler suffices.

Consequence, now triple-confirmed (seam floor, scheduler closure, and
this inventory): there is nothing left to steal in FFTW's scheduler.
The steal target is fftgen's dag construction. Minor tangent parked
for some idle afternoon: a -variables-style tiny-scope emission mode as
an alternative gcc interface to our fences; cheap to A/B, unknown
payoff. Campaign queue unchanged otherwise: streaming dag construction
(the project), GPR addressing lane (cheap), Tier 1 OOP monolithic SU
wiring (cheap).

## 32. The knob-space sweep: defaults validated, one find (PIN_FORCE)

Framing (Tugbars): the dag construction is ours, it WINS the real races
(1.21-1.80x over FFTW at plan level on avx512), and it stays. Spill
minimization is a search over the existing knob space, not a topology
redesign. Sweep: factorization (VFFT_CT_FACTOR) x scheduler
(su/bisect/bb) x regalloc force, avx2 n1, vec/gpr spill counts, all
generated and compiled identically within the sweep (fresh numbers
drift a few percent from section 29's cg_pow2-era objects; in-sweep
comparisons are the valid set).

R=16: def-su 83/35 = bb 83/35; bisect 109/32; (2,8) 87; (8,2) 93.
R=32: def-su 196/138 = (4,8); bb-5s 200; (2,16) 218; (16,2) 525.
R=64: def-su 584/322 = (4,16); bb-5s 996; (16,4) 1179.

Verdicts: (1) The hand-tuned factorization table is AT the sweep
optimum; big-first splits are catastrophic. (2) BB engages on the
spill path but never wins; at R=64 its 5s incumbent is 1.7x worse than
the list scheduler, and a 60s budget exceeded practical codegen time in
this container. Parked. (3) PIN_FORCE, the linear-scan regalloc forced
onto avx2 n1 (production gate restricts it to log3+avx512+R<=32, set
when the allocator was new and never re-tried here): vec spills
196->166 (R=32) and 584->491 (R=64), gpr down too, output BIT-EXACT.
Same-binary micro-race on this host (Cascade Lake VM, not the i9
target): +3.2 percent at R=32, noise at R=64. Lesson recorded: spill
counts overstate runtime impact when traffic is L1-resident; counts
rank candidates, races decide.

ACTION ITEM for the Phase 6 i9 audit: A/B the regalloc gate widened to
avx2 n1 R>=32 on real hardware; flip the gate if the i9 confirms.
Otherwise the knob space is now charted: no order-of-magnitude spill
reduction exists inside it, consistent with the seam floor, and the
construction philosophy stands undisturbed.

## 33. Construction audit: the dag-building layer examined for room

Map (dft.ml, 1848 lines): pick_algorithm table + VFFT_CT_FACTOR
override; Direct primes via explicit conjugate-pair factoring (stages:
pair sums/diffs, linear-chain weighted sums, paired outputs), the
family where we BEAT FFTW; dft_ct recursive CT; dft_expand_n1_blocked /
dft_expand_twiddled_spill (the doc-58 two-pass recipes with total
marker capture); twiddle policy layer; SR path (untuned, no blocked
variant).

Candidates examined, with verdicts:

A. MARKER TOTALITY (all pass-1 bins round-trip scratch): analyzed,
near-wash. Pass-2 cluster k2 consumes one bin from EVERY pass-1
cluster, so keeping any cluster register-resident trades reload
savings for equal-or-worse long-range pressure. The all-spill design
is close to in-family optimal. No action.

B. MARKER PLACEMENT / TWIDDLE SIDE (the live candidate): markers
capture PRE-twiddle values, so the internal-twiddle cmul executes
post-reload on the pass-2 side, and the boundary blocks fma_lift from
fusing it into pass-1 sub-DFT tails. Implemented post-twiddle capture
as an env-gated variant (VFFT_TW_PRESPILL); default-path regression
byte-identical; the variant itself GENERATES ILLEGAL CODE
(use-before-def in pass 2): a heterogeneous marker set (raw exprs for
trivial bins, cmul exprs elsewhere) violates an emit_c classify_passes
assumption. REVERTED for hygiene, post-revert regression byte-identical.
Candidate stands with a known blocker: classifier support for
mixed-depth markers, estimated half a day, payoff = cross-boundary FMA
fusion plus pass-pressure rebalancing. Queue after the i9 audit.

C. STORE PLACEMENT: fear disproven by measurement. Output stores begin
at line 388 of 666 (R=32) and interleave per pass-2 cluster; no
terminal batching. FFTW's stores start earlier in their files only
because their construction retires outputs before all sub-DFTs
complete, the known streaming property, not an emission defect of ours.

D. MULTI-LEVEL BLOCKING for R>=128: single-level recipe means a
128-value seam for the large_pow2 family. Real but low priority
(opt-in family, rarely planned).

E. SR-BLOCKED PATH: absent by design (source comment marks it
follow-up); SR currently loses on op count and peaks regardless
(section 28). Skip.

Net verdict: the construction is largely vindicated by its own audit.
Primes are strong, the blocked recipe's choices survive scrutiny (A,
C), and the in-family improvement room is concrete and small: B
(blocked, half-day), D (low priority). The intrinsic CT seam remains
the floor, as established, and remains a deliberate architectural
trade rather than an oversight.

## 34. Narrowing within the construction: the designed/overflow split and two stacking finds

Question (Tugbars): within the given dag construction, can spills be
minimized; can the dag be made effectively narrower? Answer: the
ALGEBRAIC width is fixed by CT, but the REGISTER-RESIDENT width is
what the recipe controls, and it moved 33.6 percent today.

THE REFRAME. For blocked codelets, counted vec movs = DESIGNED seam
traffic (the deliberate L1 round-trip; the architecture working) +
OVERFLOW (gcc pressure spills; the schedule-quality signal).
Decomposed, in-place avx2:

| R | designed | overflow | FFTW total |
|---|---|---|---|
| 32 | 130 | 66 | 65 |
| 64 | 258 | 326 | 179 |

At R=32 our overflow EQUALS FFTW's entire count: the list scheduler is
already FFTW-grade on involuntary spills there, and the visible gap is
the seam trade, working as designed. At R=64 overflow localizes by
line-attributed histogram (gcc -g + objdump -dl): pass1 carries 250 of
326, pass2 only 76. Pass-1 of CT(8,8) is the next concrete target;
diagnostic queued: windowed per-region histogram to determine uniform
vs clustered.

FIND 1 (LANDED, env-gated VFFT_N1_BLOCK_MIN, default 25 untouched,
regression byte-identical): per-ISA blocking threshold. R=16 avx2 is
monolithic only because the doc-58 threshold (n>=25) is not ISA-aware,
while its monolithic peak is 35 live on 16 registers. Blocked at
CT(4,4): total vec movs 83 to 62, overflow component ~ZERO (DFT-4
cones fit registers entirely; the codelet becomes pure designed
traffic), arith unchanged (144), bit-exact, same-binary race +16.5
percent (735 to 631 cycles) on the container host.

FIND 2 (STACKING): VFFT_PIN_FORCE + VFFT_N1_BLOCK_MIN=16 at R=16:
+33.6 percent (743 to 556), bit-exact, vec movs 61. Pin's contribution
here is allocation quality at equal counts, the counts-rank-races-
decide lesson demonstrated in one row.

i9 A/B list is now three-deep for R=16-class avx2 leaves: pin, block,
pin+block, against the monolithic default. Remaining in-family levers,
cataloged: pass-1 overflow at R=64 (localized, diagnostic next);
candidate B (classifier blocker, section 33); reload-folding audit
(gcc partially folds scratch reloads into memory operands; quantify);
the GPR addressing lane (section 29).

## 35. Defaults flipped: zero env vars for production

Tugbars's call, two grounds accepted: (1) the house style is
auto-rules, not flags; a per-ISA decision belongs in the generator, and
should_block_n1 already received vec_regs and ignored it; (2) his
hardware results have historically matched container direction, both
changes are bit-exact and reversible by regeneration, and the kernel
margins (+16.5 / +33.6 percent) dwarf plausible uarch deltas. The
provenance question, answered for the record: VFFT_PIN_FORCE was
PRE-EXISTING (doc-28-era debug override; yesterday's sweep merely ran
it on avx2 n1 for the first time); VFFT_N1_BLOCK_MIN was written
during the section-34 experiment.

FLIPPED INTO SOURCE:
- dft.ml should_block_n1: threshold = 16 when vec_regs <= 16, else 25.
  VFFT_N1_BLOCK_MIN retained as bidirectional override/back-out.
- emit_c.ml regalloc gate: + (is_avx2 && is_n1 && radix >= 16),
  evidence-exact (t1/primes unmeasured and unchanged).
  VFFT_NO_REGALLOC remains the kill switch.

IDENTITY PROOFS: avx512 n1 16/32/64 byte-identical to production
sources; avx2 n1_8 byte-identical (below both gates); avx2 n1_16 new
default EXACTLY equals the measured pin+block artifact (+33.6 percent
kernel). Back-out verified: env-restored generation reproduces the
pre-flip artifact byte-identically.

REGENERATED under flipped defaults: in-place avx2 n1 16/32/64 fwd+bwd
(cg_pow2), OOP avx2 n1 set (n1_16 now blocked; OOP unaffected by the
regalloc flip, no install_alloc there). ALL GATES PASS.

PLAN-LEVEL RACE, same-session old/new (back-out env set vs new set):
all seven cells within the host noise floor, which the UNTOUCHED
N=169 cell pins at +-4-5 percent per run (FFTW's own cycles drifted up
to 31 percent between the two runs). Honest verdict: kernel wins are
solid (same-binary, bit-exact, min-of-rounds); the plan-level effect
is real arithmetic but below this 1-vCPU VM's measurement floor; the
i9 audit VERIFIES rather than gates, per the new policy.

OPERATIONAL STATE: a full production regeneration now requires ZERO
environment variables. Remaining env vars are experiment overrides and
back-outs only: VFFT_N1_BLOCK_MIN, VFFT_NO_REGALLOC, VFFT_PIN_FORCE,
VFFT_CT_FACTOR, VFFT_SPLIT_RADIX, VFFT_COLLECT_M, VFFT_DEEP_COLLECT.
The --bisect/--bb CLI flags select non-production schedulers
explicitly. Nothing silently degrades if an env var is forgotten.

### 35b. Baseline correction and the final kernel latency table

Caught during the head-to-head: the back-out env (VFFT_NO_REGALLOC=1)
kills FENCES as well as regalloc (opt_out gates both), so the
regenerated "old" R=32/64 objects were unfenced, not true production
(R=16 unaffected; fences were off there anyway, hence its byte-identity
check passed). True-old baselines = the surviving pre-flip sweep
artifacts (fenced, verified by spill counts).

Final kernel table, same-binary, min-of-rounds, alternating order,
bit-exact everywhere (two independent runs shown):

| R | mechanism | speedup | vec spills old->new |
|---|---|---|---|
| 16 | block+pin | 1.278 / 1.327 | 83 -> 61 |
| 32 | pin only | 1.032 / 0.972 | 196 -> 166 |
| 64 | pin only | 0.994 / 1.004 | 584 -> 491 |

CORRECTED CONCLUSIONS: R=16 is robustly +28-33 percent (block alone
+16.5; pin adds ~+15 on top). R=32/64 pin is ZERO WITHIN NOISE on this
host (the two R=32 runs straddle 1.0; the binary-layout lottery is
+-3 percent); the earlier "+3.2 percent at R=32" claim DOES NOT
SURVIVE the corrected baseline and is withdrawn. Pin's spill
reductions at 32/64 are real in counts and runtime-invisible here.
Gate disposition: keep the flipped regalloc gate at R>=16 (R=16
proven, 32/64 bit-exact and never worse beyond noise, cleaner
allocation, may resolve positive on the i9's different ports);
the i9 audit carries the 32/64 verdict. Back-out note for future
sessions: VFFT_NO_REGALLOC is NOT a clean old-behavior restore for
fenced codelets; use pre-flip artifacts or git for true baselines.

## 36. OOP allocator attempt (reverted) and the guru leaf race

ALLOCATOR EXTENSION ATTEMPT: wired Emit_c regalloc into codelet_oop
(install over each render section's emission order; render_node_def
already consults the global). Spill counts responded (n1_16 137 to 99)
but PLAN GATES FAILED in both configurations tried (all sections;
pass-1 excluded): the pinned declaration form collides with the OOP
spill path's forward-declaration / no_declarator and pass-2
flush/reload conventions. REVERTED WHOLESALE; revert verified
byte-identical to the shipping set; ALL GATES PASS. Parked with a
precise blocker: render-form audit (respect no_declarator under pin;
audit pass-2 variable conventions) before retry. The blocking win at
OOP n1_16 (today's threshold flip, 226 to 137 spills) is live and
gate-passing.

GURU LEAF RACE (Tugbars's request): ours OOP n1 vs FFTW
guru_split_dft, identical column shape (element stride K, contiguous
transforms), K=512, same-binary, PATIENT-planned FFTW, machine
precision verified:

| R | old | new (shipping) | FFTW split | new-vs-old | new-vs-FFTW |
|---|---|---|---|---|---|
| 16 | 42509 | 40817 | 85785 | 1.041 | 2.102 |
| 32 | 92485 | 92595 | 211105 | 0.999 | 2.280 |
| 64 | 246547 | 247709 | 458159 | 0.995 | 1.850 |

(cycles/call, lower better; R=32/64 codelets byte-unchanged by the
flips, and their 0.995-1.003 readings certify the harness at +-0.3
percent, far tighter than the plan bench.)

Today's help at the OOP leaf: +3-4 percent at R=16 (the 39 percent
spill cut converts modestly; the K=512 leaf shape is memory-stream
dominated), zero at 32/64 as predicted.

THE FORMAT FINDING (from cross-checking the plan bench): plan-bench
FFTW (interleaved plan_many) does N=64 K=512 in ~158K cycles; guru
SPLIT FFTW needs 458K for the same work. FFTW is ~3x slower on split
arrays than on its native interleaved format. Two honest framings
follow: (a) on OUR native format, split arrays, our leaves beat FFTW
2x at every radix, the split dividend measured directly, and this is
the comparison a split-format user experiences; (b) on FFTW's native
interleaved format, its planner+execution at the same (N,K) remains
ahead of our plan-level numbers (0.6-0.9), and THAT gap, per this
race, is NOT leaf codelet quality: our own Bailey 4x16 plan (217K)
also beats our single 64-leaf (248K), so the residual avx2 deficit
lives in plan structure and second-stage execution, not in the n1
codelets this campaign has been polishing. The avx2 question, if
pursued further, moves to the t1p stages and plan shapes.

## 37. Provenance stamps: every codelet self-documents

Tugbars's request: every generated file carries, in its header, the
exact flags used and the reason for each decision. Implemented at the
GENERATOR level (genfft's pattern), so provenance can never drift from
behavior: both emitters (emit_c.emit_codelet and
Codelet_oop.emit_codelet) stamp a PROVENANCE block built from the
ACTUAL computed booleans of that emission: the literal command line,
active VFFT_* env overrides, family, ISA register file, scheduler (with
the list-scheduler naming note), GH state and its auto-rule,
construction (blocked CT NxN with the ISA-aware threshold, or
monolithic, with reasons and doc/section citations), regalloc gate
state, fence state with the n1-avx2-8/16 exemption, and log3.

Verified: comments only; disassembly byte-identical on in-place avx2
R=16, OOP avx2 R=32, and OOP avx512 R=32 against production caches.

RESTAMPED in tree: codelets/n1_oop_avx2 (22, via script),
codelets/t1p_avx2 (7, via script), 18 loose avx512 files (commands
reconstructed from filenames: n1/t1p/t1p_log3/t1s OOP + spec via
regen_spec_r32.sh + radix64 n1 in-place avx512).

OPEN (asked, not guessed): codelets/inplace_avx2 (the 216-file
odd-radix in-place set) plus the n1_oop and t1p_ext directories have
no regen scripts in-tree; restamping them means reconstructing
per-family flags for every file. Question to Tugbars on record: were
these generated exactly via generate_codelets.sh recipes (then a
restamp script follows mechanically), or do any carry special-cased
flags?

### 37b. Full-tree restamp complete

Tugbars confirmed all sets follow the generate_codelets.sh recipes.
n1_oop/ and t1p_ext/ already had regen scripts in-tree (earlier listing
truncated); ran them. Wrote the missing
codelets/inplace_avx2/regen_inplace_avx2.sh (18 families x 12 radices,
recipes mirrored verbatim from generate_codelets.sh lines 223/291),
216/216 generated. Object-identity spot checks: three avx2 in-place
families (n1, t1_dif_log3, t1s_dit) and avx512 OOP, disassembly
IDENTICAL to production caches. Every .c in codelets/ now carries the
provenance header; scripts/README.md points at all six regen scripts.

### 37c. Tree completion: pow2-avx2 and avx512 in-place sets committed

Tugbars's catch: the tree carried only the odd-radix avx2 in-place set;
the pow2 avx2 codelets and the ENTIRE avx512 in-place set existed only
in the container cache (/tmp/cg_pow2). Committed both with regen
scripts mirroring generate_codelets.sh: inplace_pow2_avx2 (6 radices x
18 families = 108) and inplace_avx512 (18 radices x 18 families = 324),
432/432 generated, all provenance-stamped. Identity spot checks, six
family/ISA combinations: four IDENTICAL outright; two avx2 t1 cases
initially DIFFERED at the object level, root-caused same session to
ZERO source drift (residual diff after comment normalization: 0 lines)
plus heterogeneous compile flags in the old /tmp cache (the cache .o
does not match its OWN source at the standard flags; recompiling both
sources identically gives byte-identical disassembly). The tree
sources are code-identical to production everywhere checked; the
/tmp object caches carry mixed-vintage gcc flags and the in-tree
scripts are now the canonical regeneration path. Tree codelet census:
736 files, 736 stamped. The tree is now self-contained: every production codelet
regenerates from in-tree scripts with zero environment variables.

## 38. Codelet tree restructure: two axes, four dirs, four scripts

Tugbars's design: codelets/{inplace,oop}/{avx2,avx512}. Adopted with
one refinement (filenames preserved; ISA suffix redundant with path
but renaming would churn history for zero gain). The "t1p avx512
missing" perception resolved during inventory: it was SCATTERED, not
missing ({4,16,32,64} loose + {7,8,13} in t1p_ext = the same radix set
as avx2). The restructure also removed five duplicate files (four
loose avx512 n1 duplicating n1_oop entries; radix64_n1_inplace_avx512
duplicating inplace r64_n1_fwd). Spec recipes unified: RV=R*8 covers
the 7/13/32 spec family with one formula.

Final state: inplace/avx2 324, inplace/avx512 324, oop/avx2 36 (22 n1
+ 14 t1p), oop/avx512 47 (22 n1 + 14 t1p + 2 t1s + 9 spec) = 731
files, 731 stamped. Each leaf dir has ONE regen.sh proven to
regenerate its complete contents (counts verified by running all
four). Identity spot checks across all four dirs vs production caches:
IDENTICAL. Build references audited before moving: everything that
mentions codelets/ paths targets the generator's scratch-output
convention, not the committed tree; nothing broke. "Is anything
missing" is now answerable per directory by running one script and
counting.

### 38b. One regeneration script for the whole tree

Tugbars's call: four per-dir scripts are four things to remember.
Replaced by a single codelets/regen.sh holding ALL recipes inline,
with target selection (default all; quadrant names for subsets). The
per-dir regen.sh files are gone; the single script proven by full run:
731/731 generated, 0 failures, all stamped, identity spot checks vs
production caches IDENTICAL. One ergonomics bug caught during the
proof: the first version used ( cd ... ) subshells which swallowed the
ok/fail counters (files regenerated, counts read zero); replaced with
pushd/popd. The tree's operational interface is now: one script, one
optional GEN variable, nothing else.

### 38c. All tooling under generator/scripts

Tugbars's call: generators belong in generator/scripts, and the cost
model (CPE measurement, memory-boundedness, plan scoring) is generator
tooling too. Moved: codelets/regen.sh -> scripts/regen_codelets.sh
(repathed, proven by quadrant run from the new home, 36/36); and
generator/cost_model -> generator/scripts/cost_model with ROOT
resolution bumped one level in all three build scripts ($ROOT/
cost_model self-references now $ROOT/scripts/cost_model; build_tuned
output location unchanged at generator/build_tuned), bootstrap.sh's
seven references updated, ps1 usage examples updated. Consumer audit
before moving: the only outside reference to cost_model was a doc
comment in core/exhaustive_screened.h. FLAGGED FOR THE i9:
run_measure_cpe.ps1 hardcodes the old Windows prototype checkout path
in its WSL command; needs the new repo path on that machine. A
codelets/README.md now points newcomers at the regen script.

### 38d. arsenal.sh: tiered entry point

Tugbars's call: full-arsenal users (codelets + CPE + memboundness)
get one wrapper; codelets-only users never touch the long
measurements. generator/scripts/arsenal.sh with stages codelets
(pass-through to regen_codelets.sh, quadrant args supported) / tools
(bootstrap.sh, which builds the scratch tree, headers, measure_cpe and
score_and_time, plus the memboundness build bootstrap lacked) /
measure (RUNS measure_cpe and measure_memboundness taskset-pinned,
CPU=n, output to generator/build_tuned) / all. Proven: fast tier live
(oop-avx2 quadrant 36/36 through the wrapper), full wiring via DRY=1.

### 38e. Strided family committed, two generator bugs fixed on the way

Tugbars's request: generate the --strided family (Design C, 2D row
FFT: B=vec_width rows at runtime row_stride, AOS->SOA transpose
preamble + inverse-transpose postamble, n1-only single-stage in v1).
Committed at codelets/strided/{avx2,avx512}, regen targets
strided-avx2/strided-avx512 (arsenal inherits). Coverage extends
production's {16,32,64}: avx2 {4,8,12,16,20,32,64}, avx512
{8,16,32,64}, fwd+bwd = 22 files. Tree total: 753.

The first regeneration FAILED TO COMPILE, all 22, exposing two bugs:

BUG 1 (latent, pre-existing): the doc-58 auto-blocking recipe gate had
no strided exclusion, so any strided codelet at a blocking-eligible
size generated spill_re/spill_im marker stores the strided emitter
never declares. Latent since the rule landed (production avx512
strided 32/64 would misgenerate too); surfaced now because strided
was never regenerated post-rule. FIX: not strided in both the
recipe-applicable gate and the blocked-expansion arm (gen_radix.ml);
strided is single-stage by design.

BUG 2 (mine, today): the widened regalloc gate keys on is_n1 by
symbol name; strided names contain _n1_ so they got the pinned render,
which uses load conventions (in_re, K) the strided signature does not
provide. FIX: not strided in the gate (emit_c.ml).

BUG 3 (latent, pre-existing, the actual compile blocker after 1-2):
three render_node_def call sites omitted ~strided while every sibling
passes it, so loads rendered with the OOP base names instead of the
transpose preamble's lane_re_j variables. A missed-call-site refactor
bug, invisible until regeneration. FIX: ~strided added at all three.

VERIFIED: all 22 compile clean both ISAs; NUMERICAL GATE vs naive DFT
at non-trivial row_stride (N=16, me=8, stride=20): machine precision
PASS on avx2 and avx512; production regression across all four
quadrants' representatives: byte-identical modulo the provenance
command path. The extended-coverage radices are generated and
gate-passing but not raced; flag for the i9 if DSP use materializes.

### 38f. Cost model split: scripts in scripts, logic in cost_model

Tugbars's correction of his own earlier phrasing: he wanted the
cost-model BUILD/RUN scripts under generator/scripts, not the logic.
Split accordingly: generator/cost_model/ holds the sources
(measure_cpe.c, measure_memboundness.c, score_and_time_plans.c),
factorizer.h, the generated/ profile headers, and the tool docs;
generator/scripts/ gains the three build scripts plus
run_measure_cpe.ps1 flat. Repathed: ROOT one level shallower in the
build scripts, source references back to $ROOT/cost_model, bootstrap's
build and generated-header paths, arsenal's memboundness call, ps1
usage comments. Verified: bash -n on all four shell scripts, arsenal
DRY run through the new paths, zero stale references outside the
README (now fixed).

### 38g. plan_executors / registry emitters: located, relocated, made reproducible

Tugbars asked where the plan_executors.h / registry*.h generators live
and how they ran. Located: bin/emit_registry_h.ml (registries) and
emit_executor_h.ml, which sat alone in emit_tool/ with a stale usage
comment; relocated to bin/ alongside its siblings, dune updated.
Registries regenerate byte-identical.

The smoke test then exposed a REPRODUCIBILITY HOLE: plan_executors.h
was generated from a spike_wisdom.txt that was never committed.
Reconstructed it from the header's forward-entry comments (15 entries;
backward blocks are variant-independent by design and carry no
variants line, which the first parse missed, and the first
reconstruction also missed 8 entries and was caught only by symbol-set
comparison after I had prematurely overwritten the tree header,
restored from the tarball, reparsed). Verified: regeneration
reproduces ALL 54 executor symbols exactly. The tree header is now
refreshed with the CURRENT emitter (the committed one predated the
cf0 leg-0 common-factor logic in the macros, i.e. it was stale
relative to the emitter); core/demo builds against it with gcc-13.
spike_wisdom.txt is committed with a provenance note. One emitter run
emits BOTH ISAs (avx512 guarded by __AVX512F__; --isa vestigial).
README carries the full recipe.

### 38h. Emitted headers wired into dune (promote rules)

Tugbars's question: are the registries part of the codelet-generation
dune flow? They were NOT: dune built the emitter executables only;
emission happened solely in bootstrap phase 2, and regen_codelets.sh
never touched them, so coverage changes could silently strand stale
registries. Fixed with generator/generated/dune: (mode promote) rules
for plan_executors.h (deps: emitter exe + spike_wisdom.txt) and both
registry headers (deps: emitter exe), so `dune build` regenerates and
promotes them into the source tree on any emitter or wisdom change.
Verified: promoted outputs byte-identical to manual emitter runs;
rebuild fires on wisdom touch; registry.h (hand-written) untouched by
construction. Residual drift risk documented: radix coverage is
defined twice (emit_registry_h.ml and generate_codelets.sh); a
coverage change must update both.

## 39. Coverage single-sourced; the tree generator goes in-process

Tugbars's call (upgrading my manifest proposal): wire everything to
the OCaml pipeline, scripts become invokers. Executed in five
identity-gated stages:

A. lib/coverage.ml — THE definition: radix sets, the 18-family matrix
   as code, OOP recipes with the RV=R*8 spec formula, strided sets,
   filenames, exact per-codelet argv. Gate: per-quadrant counts
   324/324/36/47/14/8 = 753 exact.
B. emit_registry_h.ml consumes Coverage. Gate: both registry headers
   byte-identical through the switch (the 38h promote rules fired the
   regeneration automatically).
C. gen_radix's 764-line body moved to lib/gen_main.ml as run(argv): 87
   Vfft_v2 qualifications stripped, Emit_c.provenance_argv override
   added so in-process callers stamp the logical per-codelet command;
   bin/gen_radix.ml is now a 1-line wrapper. Gate: CLI output
   byte-identical through the move.
D. bin/gen_set.ml — the driver: walks Coverage, captures per-codelet
   stdout by dup2-ing the target file over fd 1 around each
   Gen_main.run, Fun.protect-restored, fail-loud.
E. THE GATE: full tree into a fresh root, all 753 files compared
   against the committed tree: zero missing, zero content mismatches
   modulo the Generated-by line. Wall time NINE SECONDS vs ~25-35
   minutes for the 753-fork bash loop (~200x; fork + runtime init was
   nearly the entire cost).
F. Real tree restamped with logical commands; regen_codelets.sh is a
   5-line invoker; arsenal unchanged. Registry-vs-tree drift is now
   impossible by construction (one definition, dune-watched).
   Deliberate residual: generate_codelets.sh's scratch-tree lists
   (cost-model toolchain) noted in README, convertible later.

## 40. Measurement preflight: the Q3 prelaunch screen

Tugbars's design, his reference being the Quake 3 Arena prelaunch
console: not a modal warning dialog (which users click through) but a
LIVE dashboard the user watches while closing programs, fields
refreshing in place at 1Hz until the machine goes quiet, then it hands
off to the measurement run. Implemented as
generator/scripts/preflight.sh in pure bash + ANSI (alternate screen
buffer, redraw-in-place; whiptail rejected as modal-only), wired as
the gate in arsenal's measure stage.

Live fields: load, top CPU consumers by name, frequency governor on
the pinned core, SMT-sibling busy percent, max temperature, each
degrading to n/a where the host lacks the interface. Verdict line:
READY (green) / WAIT (amber, naming the blockers). Keys: enter =
launch now, a = auto-launch with 3s countdown once READY (any WAIT
resets it), q = abort. Non-interactive: --yes / YES=1 prints one
snapshot line and proceeds; no-TTY without it refuses with
instructions (exit 1). On launch a preflight snapshot sidecar
(timestamp, load, governor, sibling, temp, top consumers, host) is
written next to the measurement output: measurement provenance,
completing the stamp-everything philosophy.

Tested by driving the live screen through a pty: two bugs caught and
fixed before commit. (1) The verdict flapped READY/WAIT on the
instantaneous runnable count, jitter, not signal; runq is now
display-only and the 1-minute load carries the verdict. (2) EOF stdin
made the loop spin instead of ticking; non-timeout read failures now
sleep explicitly. Final pty runs: auto-launch counts 3-2-1-0 and
launches with snapshot written; q aborts with exit 1 and no snapshot;
no-TTY refusal exits 1.

### 40b. The literal GUI: raylib prelaunch window

Tugbars wanted an actual window. Options table considered (tkinter,
zenity/yad, Qt, GTK, Go/Fyne, Electron, PowerShell+WPF, C+raylib);
his pick: raylib, the immediate-mode game-style choice, the only one
that feels like a Q3 prelaunch screen rather than a dialog, and in the
project's native language. generator/scripts/preflight_gui.c (~300
lines): 1Hz probes (identical thresholds to the TUI, /proc and /sys
read directly in C, sibling busy from /proc/stat deltas), READY bar
that pulses green, WAIT bar in amber, mouse buttons + enter/a/q keys,
auto-launch countdown that resets on any WAIT, same snapshot sidecar
and exit codes. Built by build_preflight_gui.sh (raylib 5.0 release
fetched pinned, statically linked, ~1MB binary, runtime deps X11/GL
only, which WSLg provides on the i9). preflight.sh is now the
dispatcher: display + binary = window; SSH/no-display = the terminal
screen; PREFLIGHT_TUI=1 forces it; YES=1 unchanged.

Verified headless under Xvfb: autotest launch path exit 0 with
snapshot, abort path exit 1 without, dispatcher routes to the GUI when
DISPLAY is set and falls back cleanly without it; live window
screenshot captured and reviewed (one glyph bug caught: the default
font rendered an em-dash as '?', fixed to a hyphen, which the
project's no-em-dash rule demanded anyway).

## 41. Post-rewiring regression: in-place and OOP vs MKL

Spot-check requested by Tugbars to verify the section 38/39 rewiring
end-to-end, deliberately small: 4 pow2 cells per ISA for in-place, the
standard cell list for OOP. Everything compiled FRESH from the
regenerated tree (648 in-place + 83 OOP codelets, zero errors; avx2
binary verified zmm-free).

In-place: PASS. vfft_ns reproduced or beat the reference CSVs on all
8 cells, both ISAs (-25% to +7%); all ratios above 1. Ratio deltas vs
the doc are MKL-side cross-run drift (MKL 20-46% faster than its
reference numbers, consistent across both legs), the variance the doc
itself disclaims. Logged in wisdom_vs_mkl_regression.md section 10.

OOP: PASS, on stronger evidence than ratios. No committed reference
CSV exists, but the driver carries elementwise correctness gates vs
MKL: every natural-order cell passed at 6e-15..8e-15 and MODEB
bit-exact vs the in-place dataflow. Speed 1.40-2.29x over MKL across
all benched cells. N=2310 K=32 skipped (no plan with the 4-cell mini
wisdom standing in for the lost wisdom_v198; not a wiring symptom).

Two findings:
1. BUILD RECIPE CORRECTION (also in regression doc section 10): avx512
   builds must now link avx2 codelet objects too. plan_executors.h
   emits both ISAs with only avx512 guarded, and avx2-referencing
   executor entries postdate the original sweep.
2. LATENT, pre-existing (proven by the 753/753 identity gate, not a
   rewiring break): radix{16,64}_t1s_oop_avx512.c embed a NON-STATIC
   copy of their n1 helper, so linking them with the n1 files is a
   duplicate definition. Worked around with first-def-wins (duplicates
   identical); proper fix queued: codelet_oop should emit the embedded
   helper static.

## 42. Why we win: the answer

Joint research session with Tugbars on WHY VectorFFT wins despite
op-count parity and more spills. Three experiments, all predictions
pre-registered.

E1, instruction-class census (static, both avx2): FFTW's n1fv codelets
spend 20.9-28.7% of their vector stream on shuffle-class instructions
(perm/shuf/unpck/blend), the price of vectorizing inside one
interleaved transform. Ours: 0.0% at every radix, because batch-axis
vectorization makes every lane an independent scalar FFT. Prediction
(<2% ours, >15% theirs): both hit.

E1b, K-sweep at fixed N=1024, fixed plan: ratio vs MKL-on-our-shape is
NON-monotonic in working set: 1.26x at K=8 (overhead regime, one SIMD
group), peak 2.91x at K=64 (L2, kernel-bound: the shuffle-free stream),
dip 2.04x at K=256 (L3, unexplained, parked), rising 2.54x at K=1024
where our time scales x3.92 per 4x data (streaming) vs MKL's x4.88
(superlinear). My monotonic-fall prediction from the day-before cells:
FALSIFIED, that gradient was an N-confound. Tugbars's traffic-economy
claim confirmed as the DRAM-regime mechanism. Traffic arithmetic: our
K=1024 time = ~19 GB/s at 2 passes; MKL's implies ~5 passes' traffic
or 40% bandwidth.

E2, the discriminator (same binary, same run): MKL home shape
(interleaved, unit stride, distance N) vs MKL on our shape (REAL_REAL,
strides {0,K}). Adapter tax 3.11-3.71x, growing with K; home scaling
into DRAM x4.09 (linear) vs split x4.88. So the superlinear blowup is
THE LAYOUT, not MKL's pass structure. Both incumbents now priced on
our shape: FFTW ~3x (guru), MKL 3.1-3.7x.

THE SYNTHESIS: we did not win FFT in the abstract, we won the shape.
MKL-home is FASTER than our engine on our shape at every K tested
(e.g. 2.15ms vs 3.43ms at K=1024). Batched split-complex is a
different problem; we built the only native engine for it (zero-shuffle
batch-axis SIMD, L1-resident lane groups, 2-pass streaming), and
incumbents serve it through ~3x adapters that rot to ~5x at DRAM.
For split-native applications (the trading stack) the win is real
end-to-end: MKL-home + riffle conversion still loses for a split
deliverable. Public claims should carry the shape qualifier: fastest
engine for batched split-complex transforms.

Open doors: the K=256 L3 dip; an interleaved-native codelet family to
contest the home turf (census predicts we would eat the same ~25%
shuffle stream; residual edge would be pure schedule).

E3 addendum (FFTW's home turf, same window): FFTW PATIENT
plan_many_dft, interleaved contiguous, N=1024, vs our split engine.
FFTW-home is 1.27x / 1.65x / 1.28x faster than us at K=64/256/1024
(98992/541863/2599098 ns vs our 125715/893970/3335944), consistent
with the old avx2 N=64 anecdote (1.37x). FFTW-home wins the cache
cell outright (even over MKL-home); MKL-home leads at scale. FFTW-home
also scales superlinearly into DRAM (x4.80 per 4x data vs our x3.73),
so the home advantage compresses at the memory wall. The asymmetry
that summarizes the thread: their adapter to our shape costs 3-3.7x;
our shape costs us 1.3-1.65x on theirs, and for a split deliverable
their home time plus the riffle conversion still loses (4.3ms vs our
3.34ms at K=1024).

E4/E5 addendum: Tugbars's correction and the collapse of the
home-turf gap. E3 compared FFTW PATIENT against our STALE wisdom plan
(the regression doc's own "ratios are a floor" caveat, cashed in). Our
PATIENT-equivalent: a 116-candidate plan race at N=1024 K=256
(29 factorizations over radices 4..64 x 4 variant patterns, raced
through the verified harness, pace 0). Winner: 32x32 FT at 515234 ns,
the SAME factorization FFTW's PATIENT picks for itself, 1.51x faster
than the old wisdom's 4x4x4x4x4 (my pre-registered 10-25% improvement:
FALSIFIED UPWARD). Same-window verification at K=64/256/1024: the
FFTW-home advantage collapses from 1.27-1.65x to 1.01-1.05x, parity
within container drift; MKL-home to parity-to-1.1x. MKL-split ratio
with the proper plan: 2.75x at K=1024.

Mechanism of the stale loss, confirming Tugbars's loads/stores
suspicion: identical arithmetic, five array round trips vs two; pass
economy inside our own engine. Planner note: the winner is K-dependent
(16x64 at K=64, 32x32 at K>=256), so per-(N,K) wisdom is load-bearing.

PROMOTED TO TOP PRIORITY for the i9 audit: regenerate wisdom on
current codelets BEFORE any plan-level measurement; the shipped
wisdom is ~1.5x stale and would mismeasure everything downstream.
The why-we-win synthesis stands corrected on its last clause: on
their turf we are at parity, not behind, while delivering split.

E6 addendum: why the home-turf TIE is possible at all (Tugbars's
residual question: more ops + more spills should lose). Premise
corrections: op count is DAG-parity (campaign), and FFTW spills too:
flop-normalized census of the two plans' actual stages (per 100 vector
flops) gives vfft r32_n1 spill=19.2 / r32_t1s spill=31.6, shuffle=0
both, vs fftw t1fv_32-avx512 shuffle=28.6 spill=6.1 and n2fv_32-avx2
shuffle=25.9 spill=20.9 (their avx2 leaf is spill-PARITTY with our
stage 1). Total overhead per 100 flops: ours 19-32 (all load/store
ports), theirs 32-47 (port-5 shuffles, partly dataflow-serial, plus
spills), plus their buffered memcpy pass. Same p0/p1 FMA pressure on
both sides; each engine's baggage rides non-bottleneck resources. Net:
the measured 1.01-1.05x. Spills were always the cheap overhead class;
shuffles are the expensive one because deinterleave sits on the
critical path. Methodology note: one printf column bug caught and
corrected via raw counts; extraction loose end (libtool lt334- member
prefixes) closed.

## 43. The action plan from the why-we-win session

Tugbars's conclusion from the E-series: DAG shape and scheduler have
headroom. Discriminator experiment: VFFT_PIN_FORCE on avx512 r32_t1s
(never covered by the avx2-n1-only regalloc gate) cuts spill movs
225 -> 114 (49%; my 15-25% prediction falsified upward, the third
upward falsification of the session). Same-binary race: 1.050 / 0.985
across rounds, bit-exact, time-neutral at container resolution; my
3-8% conversion prediction unresolved. Port arithmetic on reflection:
mov pressure runs marginally under FMA pressure, so spills still hide
at L1 scale. Same verdict shape as the avx2 R=32/64 pins.

THE RANKED PROGRAM (with measured evidence per item):
0. Wisdom regeneration on current codelets (plan layer, ~1.5x stale,
   FIRST on the i9; the 116-candidate race harness is the tool).
1. Twiddle-family allocator gate (t1s/t1p, both ISAs): 49% spill cut,
   time-neutral in container, i9 A/B candidate alongside the avx2
   leaf three-deep A/B. Smallest diff; targets stage 2 of the winning
   32x32 plan.
2. Construction floor (the remaining 114 spill movs): streaming /
   split-radix DAG ordering, Candidate B post-twiddle capture, R=64
   pass-1 localization (250 of 326 overflow). Measured ceiling now
   attached.
3. GPR addressing lane (308 vs FFTW's 0): the one absolute column.
Non-targets: avx2 n1 leaves (done), planner search method (proven),
n1 blocking thresholds (shipped).

Section 43 addendum: plan-level A/B of the t1s pin (Tugbars asked for
the FFTW comparison after the change). Method lesson first: sequential
shipped-then-pinned blocks showed 9-15%, but the MKL column moved with
the vfft column (its K=1024 time improved 13% with no change on its
side), so the window drifted mid-experiment; interleaved A/B/A/B over
three rounds collapses the effect to +2.3/+2.8/+4.6% (paired mins).
K=256 within container noise; K=1024 pin faster 3/3 rounds at +4.6%,
consistent sign at the noise edge, exactly where spill traffic should
cost most. Item 1's prior upgrades from "time-neutral" to "plausibly
~4% at memory-heavy cells"; i9 A/B remains the gate. Versus FFTW-home
the parity conclusion is unchanged: pinned best 496-508K at K=256 vs
FFTW's 542-630K window range, 2.42M vs 2.15-2.60M at K=1024. The pin
is NOT shipped; it remains an env-override experiment pending the i9
verdict (flipping it = extending the emit_c regalloc gate to the
t1-family, one condition).

Final container word on the t1s pin: 8-round interleaved sign test at
K=1024: 6/8 pin-faster, median +3.2%, p=0.145, per-round swings -17%
to +19% (noisy hour, noise floor 3-4x the effect). Pre-registered
discriminator (real effect = 7-8/8): between hypotheses. STATUS:
directionally favorable, unproven, i9 gates the flip. Decision rule:
>=2% on 32x32 TT at K>=256 on quiet hardware -> extend the emit_c
regalloc gate to the t1-family; else SU stays. The withdrawn R=32
+3.2% claim is this situation's cautionary twin.

## 44. Criticizing-goggles review -> the DAG/scheduler program

Joint design review of construction and scheduler. Full backlog with
gates: docs/dag_scheduler_program.md. Highlights: phase-ordering trap
resolved by three complementary moves (spills as schedulable IR nodes
with allocator choosing WHAT and scheduler choosing WHERE; dual-
objective Goodman-Hsu-style priority, ILP far from the register wall
and liveness at it; whitelist replaced by generate-both-count-keep
stamped policy). Construction critique made accessible: the one-shape
recipe ([block]x N then [combine], uniform-depth markers) is the
32-real seam; Candidate B's mixed-depth markers are the grammar that
permits combine slices to start early (wash the bowls as you cook).
Tugbars's correction folded in and it sharpened the plan: SR's
arithmetic is no better than CT+CSE, so the prize is the narrow-
frontier EMISSION ORDER, portable to CT (A7); SR rehabilitation
demoted to reserve (A8). CT-factor portfolio search declined by
design owner, recorded. A1 (schedule-quality analyzer: critical path
+ port histogram from the uarch tables) gates all scheduler changes;
the open bet from the session stands pre-registered there: whose
critical path is shorter at r32, ours or MKL's.

## 45. A1 ships and immediately re-aims the program

sched_analyze.py (cost_model/) v1: disasm-level dependence graph
(registers exact, rsp-offset spill chains tracked), CLX latency
table, main-loop-body isolation, CP vs per-port bounds. First table:
ALL kernels, ours and FFTW's, are p01 PORT-bound (slack 0.13-0.50);
my pre-registered bet falsified both ways (our CP shorter, 8.7-10.3
vs FFTW 14.9 per 100 flops; our p01 pressure higher, 35.8-37.2 vs
32.9). Pin-vs-SU port profiles near-identical, agreeing with the
race's time-neutrality. The discovered lever: FMA fusion, 53% of our
flops via FMA vs FFTW's 68%; at a p01-bound kernel that is ~10-12%
of pressure on the binding port. A2 re-scoped accordingly (dual-
objective scheduler shelved by its own gate). v1 caveats on record:
no array store->load deps, fixed latency table, single-iteration CP.

## 46. A2 closes in one measurement: zero fusable pairs

The fusable-pair count (single-consumer vmul feeding add/sub, register
lifetime tracked): ZERO at r32_n1 and r32_t1s. Decomposition: r32_n1's
"246 mul/adds" = 6 muls + 240 bare add/subs; 140 FMAs already hold
every fusable multiply; the 6 strays are CSE-shared (correctly
unfused). Emission optimal; my pre-registered branch ("if the count is
small, the 68% comes from constructing fusable shapes -> A7") landed.
Corrections folded: the 53-vs-68 contrast was family-mismatched (n1 vs
t1fv); like-for-like 57 vs 68 (t) and 53 vs ~57 (n1). Real currency at
p01-bound kernels: binding-port uops per flop, ours 0.716-0.734 vs
FFTW 0.658 (~8-10% fewer slots on the bottleneck for the same math,
via multiply-accumulate DAG shapes CT butterflies don't produce). A7
gains the metric as a gate; A8's SR entry condition sharpened. The
critique of record lives at docs/dag_scheduler_critique.md.

## 47. A5 falsifies its way into A7

Two GPR spill classes clarified for the record: vector spills (FFT
data overflowing zmm/ymm) and GPR spills (addresses/strides/counters
overflowing the 16 scalar registers; a vector load cannot issue until
its spilled address reloads). A5 attacked the second standalone and
died twice, pre-registrations falsified (session count: 7 and 8):
the flag sweep found no compiler-side relief, and the FFTW-mimicking
root-sum transform reproduced FFTW's instruction mix exactly (imul 0,
lea 165) while WORSENING gpr spills 298->404, with an unexplained vec
improvement 323->279 (parked). Root cause: address LIFETIMES, not
representation; our emission keeps all 64 element offsets live across
the body, FFTW's wave-ordered emission keeps ~8. Conclusion: the GPR
lane is the vector seam's defect one register file down. A5 merged
into A7, which now carries three gates: liveness frontier (32-real
seam toward FFTW's 8), p01-uops/flop (0.72-0.73 toward 0.66), GPR
spills (298 toward ~0). Three instruments, one defendant: the
constructor's emission order. A6 (mixed-depth markers) remains the
door.

## 48. The committed S1 prosecution finds a different culprit

Tugbars: attack S1 committedly, and check MKL's assembly for how ILP
is done. Done, with the session's biggest practical find at the end.
(1) Validation: r32_n1 measured ~810 cy/group vs 193 port floor,
efficiency 0.24 (pre-reg 0.75-0.85 falsified, #9), reopening S1.
(2) Fence-strip: gcc free reordering, +90 vec spills, time FLAT
(#10): fences and our order both acquitted. (3) MKL kernels extracted
(mg_colbatch_plain_fwd_32_d, mg_rowbatch_twidl_fwd_032_d): def-use
spacing median 12 vs our 4; colbatch interleaves batch columns; their
static profiles WORSE (twiddle p5-bound 51.9/100fl, FMA:mul 0.4-1.3);
parity via window-filling. (4) Short-body window test: r16 efficiency
0.23 = r32's (#11): body length acquitted. (5) The suspect that
confessed: pow2 ios L1 SET-ALIASING, 2.4x by itself (305->128
cy/group at ios 256->264/320/8). (6) PRODUCTION VERIFIED: K=264 vs
K=256 on the 32x32 plan, 15.8-28.6% faster per point (pre-reg >15%
HIT). FFTW's buffered wrapper reinterpreted as the aliasing dodge.
K=256 dip plausibly explained (flagged). Program: A9 (padding/staging
escape, top tier, +16% measured) and A10 (spacing scheduler, demoted,
i9-gated) added. Scheduler exonerated for the third and final time
this session.

Section 48 addendum: the M project probe. Tugbars pointed at the M
project as the in-tree answer to window starvation. Located
precisely: emit_c ?fuse=M (interleave M pass-2 sub-DFTs), threaded
into the in-place blocked path via make_spill_info; my first grab,
VFFT_COLLECT_M, corrected on record (algsimp mult-collection,
unrelated, default-off). Probe: r32_n1 avx512 fuse=2 vs baseline at
clean stride ios=264, bit-exact, spills 103 vs 105, speedup
1.006/1.021 (<=2%; pre-reg resolved to the downclock branch). Window
starvation now triply falsified in container. A10's gate tightened;
fuse-M sweep is its first i9 implementation path before any
scheduler change.

## 49. Rader/Bluestein restored into core (integration log)

Tugbars supplied the four missing pieces: bluestein.h (the lost
header: _bluestein_choose_m, _bluestein_block_size [L2-fit, 1MB
target, SIMD-multiple B], _blue_cmul_vv, stride_bluestein_plan),
vfft_wisdom_tuned.txt (THE v5 stride wisdom, 198 entries — restored
to core/; my 4-cell CSV reconstruction verified byte-faithful against
it; contains NO prime cells by design), bluestein_wisdom.h (separate
per-(N,K) wisdom: M and B for prime cells; heuristic
_bluestein_choose_m documented as up to 4.65x off at N=179 where
M=384=64x6 beats M=361=19^2), and bluestein_calibrator.h (the (M,B)
sweep: Rader fixes M=N-1 and sweeps B in {16,32,64,128,256,4,8};
Bluestein sweeps smooth M in [2N-1, 4N] x B).

Integration gaps found and fixed (the headers were written against
PRODUCTION core; this is the prototype lineage):
1. Lineage-A stride_plan_t (defined in
   prototype/generated/plan_executors.h) lacked the override hooks
   the plan shells install. Added: override_fwd/bwd(void*,double*,
   double*), override_destroy(void*), override_data. calloc'd
   creation leaves them NULL for normal plans.
2. core/proto_stride_compat.h (NEW): the lineage adapter and the
   override-dispatch site. Maps stride_execute_fwd/bwd[_serial] onto
   vfft_proto_execute_fwd/bwd(plan,re,im,plan->K), stride_plan_destroy
   onto vfft_proto_plan_destroy, honors override hooks first, aliases
   stride_registry_t/stride_wisdom_t/stride_wise_plan to the
   vfft_proto names so bluestein_calibrator compiles unmodified, and
   provides STRIDE_ALIGNED_ALLOC/FREE. Include order: executor.h,
   planner.h, threads.h, proto_stride_compat.h, bluestein.h, rader.h.
   (Equivalent dispatch hooks also added to the standalone lineage-B
   stride_executor.h for symmetry.)
3. CRLF stripped from all four uploads on install.

GATE: Rader smoke at N=17 K=32 B=16, inner 16-point plan via
auto-plan, full pipeline vs naive DFT: maxerr 8.3e-14, PASS. Wisdom
loader verified: 198/198 entries (it returns 0 on success; first
read of "0 entries" was my misreading of the status code).

NEXT: awaiting the bluestein wisdom file (N K M B best_ns) for the
tuned prime cells, then the FFTW bench: in-place split at primes
{17, 101, 257, 1021}, K=256, vs FFTW PATIENT home-interleaved and
guru-split (FFTW runs its own Rader at primes).

### 49b. The prime bench: Rader/Bluestein vs FFTW PATIENT

Tuned wisdom installed (core/vfft_bluestein_wisdom.txt, 36 entries;
large-Rader cells 641/1009/2801/4001 deliberately trimmed by Tugbars
as noisy, heuristic fallback). benchmarks/bench_rader_vs_fftw.c: six
prime cells at K=256, our in-place split Rader/Bluestein (tuned M,B
from bluestein wisdom; inner plans from the v5 stride wisdom) vs FFTW
PATIENT on its HOME interleaved layout (its own Rader inside),
correctness-gated per cell vs the naive DFT.

Result: clean sweep. Rader: 127 -> 2.66x, 257 -> 3.12x, 401 -> 2.91x.
Bluestein: 83 -> 1.53x, 107 -> 1.85x, 179 -> 1.21x. Errors 6e-13 to
1.3e-11 (expected growth through the double-FFT convolution), all
PASS. The significance: at composite N our-split-vs-their-home is
PARITY; at primes we beat their home by 1.2-3.1x. Prime handling is
where the structures diverge most, and the factorization story shows
live: 257 (inner 256, pure pow2) tops the table, confirming rule 1
(Rader cells rank by pow2-richness of N-1; the calibrator's M choices
- 96/192/384/640, never radix-19-heavy - are the same rule applied to
Bluestein's free M; tuned B is a real knob: 257 K=256 went 264010 ->
179026 in the calibrator vs the old override run).

## 50. The trig family: wiring audit, dct.h restoration, OCaml vs python

Tugbars asked whether the May trig-family expanders still wire into
the much-changed DAG machinery, then supplied dct.h + the two python
n8 generators. Findings:

WIRING: functionally intact. gen_main's --dct2/--dct2-trigII/--dct3/
--dht/--dct4/--rdft flags route Dft_r2c.dft_expand_* assignment lists
into the CURRENT pipeline (algsimp -> CSE -> SU -> regalloc -> emit).
Numerics: dct2 PASS at N=8 avx2 (1.1e-14), N=16 (1.2e-13), N=32
(2.2e-13), N=64 (4.5e-13) avx512. Provenance-stamped. Whole family
generates clean. Three gaps: (1) ABI drift: trig emits the generic
7-arg signature; production dct.h consumes the May-era lean 3-arg
(in,out,K) - fresh OCaml output needs a lean r2r signature in emit_c
or shims; (2) not coverage citizens (no codelets/trig tree, no
registry, no regen); (3) flat-only construction, regalloc pin gate
never matches trig names (fine through N=64).

DCT.H RESTORED: core/dct.h + core/dct2_n8_avx2.h + core/dct3_n8_avx2.h
(emitted from generator/scripts/reference/gen_dct8.py and
gen_dct3_n8.py, now in-tree). Two more lineage gaps fixed en route:
stage struct gained n1_scaled_bwd (NULL-defaulted; r2c.h guards every
use), and proto_stride_compat.h gained _stride_execute_fwd_slice_from
+ _stride_execute_bwd_slice_until (parameterized copies of
executor_generic's stage loops, for r2c's fused first/last stage).
GATE: production DCT-II vs formula 2.2e-15 (N=8, python fast path)
and 1.7e-13 (N=32, full Makhoul->r2c->inner path exercising the new
slice executors); DCT-III roundtrip 3e-16 / 9e-15. PASS.

OCAML VS PYTHON at N=8 avx2, K=4096, same window: ocaml/py = 1.090
to 1.092 - the hand codelet wins 9% (my TIE +-3% pre-registration
FALSIFIED, #12; May's tie did not reproduce in this container).
Cross-output agreement 4.7e-14.

NEXT (sized): (a) lean r2r ABI in emit_c (small branch, unlocks dct.h
consuming OCaml output directly, including N=16/32 fast paths that
have no python equivalent); (b) coverage citizenship for the trig
family; (c) optional: E/O symmetric factorization in the dct2
expander if the op census attributes the 9% to construction.

Section 50 addendum, the census verdict: the 9% is NOT construction.
OCaml's algsimp'd DAG beats the hand factorization on op count (41
arith uops per 4-lane iter: fma 10, mul 14, add 17 - vs python's 58:
fma 28, mul 7, add 23). The loss is constant codegen: python hoists
all 8 constants into registers pre-loop (6 broadcasts, 5 loads, 8
stores per iter); our emission re-materializes constants inside the
loop (18 loads, 11 stores per iter). Fix is a small emit_c change:
hoist loop-invariant constants for loop-form codelets. With that, the
OCaml output should win the N=8 race outright, and it already owns
everything the python codelets cannot reach (N=16/32/64, avx512,
dct3/dct4/dht/rdft variants).

## 51. OCaml ownership of the trig family: three steps, executed in order

STEP 1, constant hoisting. NK_Const nodes were emitted as fenced
register temps inside the k-loop, forcing per-iteration
re-materialization (18 loads/iter at dct2 N=8 vs python's 5+6
hoisted). Added render_hoisted_consts to emit_c: every NK_Const
rendered once, unfenced, at function scope before the loop; in-loop
renderers skip hoisted tags; names and arithmetic order unchanged so
outputs are bit-exact. RESULT: dct2 N=8 avx2 loads 18->8 plus 6
broadcasts (python's exact memory profile) and the race FLIPS:
ocaml/py = 0.974-0.981, the generated codelet now beats the hand
codelet ~2%. GATE FINDING: the same hoist TAXES spill-bound DFT
kernels ~2% (r32_n1 bit-exact but slower: hoisted consts live across
the whole body and steal registers from data). Hoisting is therefore
GATED to the trig family via Emit_c.hoist_consts_enabled, set by
gen_main for {rdft, dct2, dct2_trigII, dct3, dht, dct4}; with the
flag off the DFT tree is byte-identical (verified: stamp-only diff).

STEP 2, lean r2r ABI. Emit_c.r2r_signature (set for the real-to-real
five: dct2, dct2_trigII, dct3, dct4, dht; rdft excluded, complex
output) switches the OOP signature to (const double *in, double *out,
size_t K) and maps in_re->in / out_re->out. GATE: dct.h running on
OCaml codelets directly (python fast paths masked via include guards,
radix8_dct2/dct3_avx2 substituted): DCT-II 1.1e-14, DCT-III roundtrip
9.3e-15, PASS.

INCIDENT, caught by the gate: prototype/generated/plan_executors.h is
dune-promoted, so the dune builds during step 1 REVERTED the hand
patches from section 49 (override hooks, n1_scaled_bwd). Fix applied
at the source of truth: generator/bin/emit_executor_h.ml now emits
both extensions; promoted header verified; full post-fix regression
(rader smoke, production dct gates) PASS. Lesson on record: never
patch promoted artifacts, patch their emitters.

STEP 3, coverage citizenship. coverage.ml gains trig-avx2/trig-avx512
quadrants: {dct2, dct3, dct4, dht} x N {8,16,32,64}, lean ABI,
hoisting on. gen_set generated all 32 into codelets/trig/{avx2,avx512}
(tree now 753+32=785), all compile. Property gates (avx512, N=8..64):
DHT self-inverse 3.6e-15..1.4e-14 PASS; DCT-IV involution
2.8e-14..5.9e-14 PASS. With dct2's formula gates and dct3's
roundtrips, all four kinds are gated. Open, on record: scalar-ISA
trig coverage, rdft citizenship, avx2 property sweep, and dct.h
gaining N=16/32 fast-path dispatch (the codelets now exist).

## 52. Trig family in the generation script + FFTW comparison

SCRIPT: proven end to end. `regen_codelets.sh all` now produces 785
files (648 inplace + 83 oop + 22 strided + 32 trig); flag-leakage
check clean (DFT codelet regenerated through the warm gen_set process
diffs stamp-only vs CLI output); tree trig files carry the lean ABI
and hoisted constants.

FFTW COMPARISON (benchmarks/bench_trig_vs_fftw.c, committed): four
kinds x N{8,16,32,64}, K=256, our avx512 trig codelets vs FFTW r2r
PATIENT on two layouts (home: contiguous transforms via
plan_many_r2r istride=1 idist=N; split: our layout istride=K dist=1).
Cross-check vs FFTW elementwise per cell: 2e-14..9e-13, ALL PASS
(the May convention work holds exactly).

Result: clean sweep, 2.9-19x.
  dct2: home 6.7/6.6/4.4/3.1x (N=8/16/32/64), split 7.4/8.8/7.1/5.3x
  dct3: home 6.3/7.7/4.8/3.8x,                split 6.5/9.9/7.5/5.8x
  dct4: home 18.4/6.4/5.6/4.0x,               split 19.5/9.0/7.8/6.3x
  dht:  home 19.0/4.9/3.3/2.9x,               split 20.8/11.4/11.2/7.8x

MECHANISM (stated so the numbers are believed for the right reason):
FFTW's r2r path does not SIMD-vectorize across the batch dimension;
its rdft codelets work per transform, largely scalar at these N. Our
codelets put 8 transforms per zmm op along K - the batch-axis
advantage at its purest, with zero shuffle tax AND zero scalar tax on
the other side's ledger. The 18-19x outliers at dct4/dht N=8 reflect
FFTW's documented second-class treatment of REDFT11 and DHT kinds.
Caveats on record: container/KVM, single window, PATIENT planning,
K=256 only; i9 confirms magnitudes. These are leaf-level numbers; a
plan-shell comparison (dct.h pipeline vs FFTW at large N) is the
separate follow-up.

## 53. The K=1 question: width cascade, scalar ISA restored, bit-exact

Tugbars: can we answer the incumbents' "must serve K=1 and
transform-contiguous data" without going interleaved, by cascading
AVX-512 -> AVX2 -> scalar when batches don't fill the lanes?

ANSWER: the constraint decomposes into two, and the cascade fully
kills the first. (A) Arbitrary K: since batch lanes never interact,
the SAME DAG emitted at widths 8/4/1 computes each lane identically;
K = 8a + 4b + tail dispatches as zmm main + one ymm pass + scalar
lanes, and the result is BIT-EXACT with the pure-vector path. SSE2
skipped: a 2-3 lane scalar tail costs nothing.

BUILT: scalar ISA restored to this lineage (isa.ml: vec_width=1,
plain-double op constructors, __builtin_fma for single-rounding
parity with vector FMA, xor_pd as negation under its verified
-0.0-only contract, +x fence constraint; regalloc maps width-1 to
xmm; emit preamble gains two load/store shims for the raw spill
sites). PoC (r16_n1, ios=64): maxdiff = 0 BIT-EXACT at K in
{1,2,3,5,7,13,14,23,64}; cost curve 91 cy/lane at K=1 -> 14.3 at
K=64 (matching section 48's clean-stride zmm figure). Total
implementation: ~60 lines of branches, one regalloc line, two shims,
nothing touched in math/algsimp/scheduler.

(B) Transform-contiguous layout and K=1 at large N: the cascade does
not address layout. The honest ledger: K=1 small-N is served
respectably by the scalar codelet (91 cy for a 16-point FFT); K=1
LARGE-N belongs to the four-step algorithm (N = RxC: C-batched
R-FFTs + twiddle + R-batched C-FFTs + transposes), which MANUFACTURES
our batch axis from a single transform - proposed, not built. The
layout question becomes a copy-cost question (a blocked transpose
adapter, ~2 memory passes, vs the incumbents' measured 3-3.7x
adapters), no longer an impossibility.

STRATEGIC UPDATE to the section-52 sentence: the K-multiplicity
constraint is DEAD (any K >= 1, bit-exact); what remains of the
incumbents' moat against us is only the layout contract, and that is
priced in memory passes, not in architecture.

NEXT (sized, on record): executor cascade dispatch (segment the K
loop per stage, registry gains a scalar column), scalar coverage
quadrants (+~370 files via the same gen_set path), four-step plan
node for K=1 large-N.

### 53b. The membrane priced: falsification #13 and the real design

Can we serve transform-contiguous users without the interleaved
route? Yes, but the measurement killed the lazy version. Form 1
(explicit blocked corner-turn round trip, re+im, 1024x256, 64x64
tiles): 1.93 ms = 199-241% overhead vs the 32x32 plan's 0.80-0.97 ms.
Pre-registration (<=25%) FALSIFIED (#13). The arithmetic is
bandwidth: the FFT at this size is ~2 passes; an explicit membrane
adds 4. And the number lands exactly on the incumbents' measured
3-3.7x split-adapter tax: the explicit-copy layout tax is symmetric
PHYSICS, not their engineering failure.

The viable design is therefore Form 2, the FUSED membrane: stage 1
loads 8-record tiles from U-layout and transposes in-register
(shuffles riding p5, which our kernels leave idle); the last stage
transposes on store. Extra memory passes: ZERO, same bytes as native,
different access pattern. Predicted overhead 10-25% (shuffle uops +
tile locality), to be measured when membrane codelets exist. Needs:
Isa gains shuffle/permute constructors + a tile-transpose emitter, or
hand-written boundary kernels feeding normal codelets.

THE PRIZE, restated: they serve our shape at 3-3.7x because they
don't fuse; with a fused membrane we serve THEIR shape at a predicted
10-25%. Whoever fuses the membrane into compute owns both layouts.
The generator is the right tool to emit it.

### 53c. Membrane vs interleaved codelets: the decision evidence

Second falsification (#14): the L2-sliced shuffle-free membrane
(stream 8 records into 128KB split scratch, run K=8 plan, stream
back; DRAM traffic native-equal) measured +219-222% vs native
(647us -> 2.07ms at 1024x256). Pre-reg <=30% destroyed. Mechanism:
not bandwidth this time but SCALAR STORE THROUGHPUT - a shuffle-free
corner-turn means stride-8 scatter stores, one double per store.
Combined with #13: every shuffle-free layout crossing measures ~3x
(explicit form: DRAM passes; sliced form: store throughput), exactly
the incumbents' adapter price. The membrane is viable ONLY with
in-register 8x8 shuffle tiles (8 loads + ~24 p5 shuffles + 8 vector
stores per 64 doubles, riding the port our kernels leave idle).
Ceiling estimate +5-15% whole-plan; held LOOSELY after today's two
falsifications - decision-grade only when the v1 kernel exists.

The comparison, stated for the record:
- Interleaved codelet family: months (new vectorization model:
  lane mapping, shuffle networks, twiddle layout, permutation
  synthesis = re-doing genfft's simd layer); ceiling = FFTW-parity
  (their home equals our measured parity); doubles the tree; uniquely
  serves K=1 small-N natively.
- Fused membrane: days for the hand-written 8x8 v1, weeks for
  generated membrane codelets (Isa shuffle constructors + transpose
  emitter); engine core untouched; predicted small tax; K=1 large-N
  via four-step (which reuses the same tiles), K=1 small-N via the
  scalar cascade.
DECISION RULE: build membrane v1 first (cheap, decisive); the
interleaved family is justified only if v1 measures above ~40%
overhead, and even then only for the K=1-small-N niche.

Section 53 closing note: the layout market survey and the
membrane-vs-interleaved decision are consolidated in
docs/layout_market_and_membrane.md (terms, decision evidence with
falsifications #13/#14, the domain-by-domain map of who lives on
which layout, and the strategy map).

## 54. Coverage inventory: DST joins, the roadmap doc

Tugbars: do we support all DSP/trig transforms? Audit found two gaps
OF OURS from this session's own work, both closed on the spot: (1)
dst.h/dct4.h/dht.h from the original upload had never been installed
to core (now installed, full r2r production chain compiles); (2)
dst2/dst3 exist in gen_main but were missing from my hoist/r2r flag
expressions AND the trig coverage kinds — added; tree now 48 trig
codelets (6 kinds x 4 N x 2 ISA, 801 total), dst3(dst2(x))/(2N)
roundtrips PASS at 1.2e-14 (N=16) and 2.0e-14 (N=64).

The full inventory, the missing-kinds list (DST-IV trivial, DCT-I/
DST-I real work, odd-N r2c), the product layers (MDCT, convolution
API, Hilbert, CZT, Goertzel/sliding, PFB), and the strategic axes
(2D/3D plan shells, float32, fused membrane) live in
docs/transform_coverage_roadmap.md with effort sizing and a
suggested order.

## 55. Tier-K plan, DST-IV executed inside the planning turn

DST-IV (RODFT11) DONE: exact reduction derived and implemented at the
math layer (dft_expand_dst4: input (-1)^n sign flips + output
reversal over dft_dct4; algsimp folds the signs into constants, so
the codelet costs exactly a DCT-IV). Wired: --dst4 flag, naming,
hoist + r2r ABI expressions, coverage (7 kinds, 56 trig codelets).
Gates: vs formula 4.6e-14 (N=8) / 1.4e-13 (N=32); involution
(2N scale) 3.0e-14 / 5.0e-14. PASS. Pending: RODFT11 row in
bench_trig_vs_fftw.

INCIDENT owned: the first naming-branch edit used a lazy regex that
matched from the dispatch block to the naming block and duplicated
~400 lines of gen_main.ml. Caught by the build; restored from the
output tarball (the repack-every-turn discipline is also the backup
discipline); redone with exact-string anchors. Rule reinforced:
multi-occurrence tokens get exact anchors, never lazy spans.

The remaining tier-K plan (DCT-I/DST-I two-phase with a
pre-registered symmetry-CSE bet, odd-N r2c two-phase) is written
into docs/transform_coverage_roadmap.md with gates and sequencing;
2D FFT noted as existing in Tugbars's codebase, integration pending
delivery.

## 56. DCT-I/DST-I phase 2: the aggressive-policy corruption hunt

Expanders landed (dft_expand_dct1: even extension M=2(N-1), Y[k] =
Re(DFT_M[k]); dft_expand_dst1: odd extension M=2(N+1), Y[k] =
-Im(DFT_M[k+1]); zeros and sign flips fold at construction). First
gates: M=8 perfect, every M>=16 wrong by O(0.1-2). The hunt, in
order, every probe on record:

1. Input loads verified: exactly indices 0..N-1, extension composes
   correctly through the CT recursion.
2. rdft-32 control fed the SAME extension from the C side:
   bin-perfect natural order. Dft.dft, ordering, constants, emit all
   exonerated for injective inputs. The one structural novelty of
   dct1/dst1: NON-INJECTIVE input maps (duplicated leaves).
3. Forced factorizations: CT(2,8) PASS, CT(8,2)/CT(4,4)/split FAIL.
4. Emit-im-too probe: im comes out EXACTLY 0.0, re still wrong ->
   corruption lives in the DAG values, emit pipeline exonerated.
5. Pass A/B by env: dedup_sub_pairs off (error changes, still fails),
   fma_lift off (no change), reassoc forced (BREAKS even N=5 - second
   latent bug, ledger #16). All default-path passes exonerated... at
   the flags I assumed.
6. Constant fingerprint: failing build contains set1_pd(2) absent
   from the passing build - a same-term fold firing on duplicated
   leaves. Pointed at the right layer, wrong pass.
7. THE INSTRUMENT: bin/dbg_eval.ml - numeric evaluation of the DAG
   after every pass vs brute DCT-I. First run (aggressive=false):
   all passes CLEAN. Mismatch found: gen_main computes
   aggressive/reassoc/fma_lift_safe from pick_algorithm(CLI n);
   N=9 is odd -> Direct -> aggressive=TRUE. Re-run with the real
   policy: factor_common_muls ~aggressive:true takes the DAG from
   5.2e-15 to 1.258e+00. Scalar end-to-end error was 1.26. Caught.

ROOT CAUSE: policy misrouting. The boundary kinds are the first
expanders hiding a CT graph behind an odd user-facing N. gen_main
classified them as Direct primes and ran the aggressive factor
passes over a CT DAG - the combination algsimp.ml's own comments
(~1418) document as unsafe ("stray same-const fires that DO pass
safety in CT"). N=5 passed by luck: CT(2,4) too small to trip it.

FIX (section-56 patch): policy_n = internal DFT size (dct1: 2(N-1),
dst1: 2(N+1), else n) drives needs_reassoc, aggressive, and
fma_lift_safe. Three sites in gen_main. dbg_eval mirrors it.

GATES after fix: dct1 N=5/9/17/33 and dst1 N=7/31 formula +
involution all PASS at 1e-13 class; N=9 green on scalar, avx2,
avx512; dst4/dct4 regression spots unchanged.

FALSIFICATION LEDGER:
#15  Aggressive factor_common_muls is value-corrupting on CT DAGs.
     Known-latent by design comment, now with a reproducer:
     dbg_eval at N=9 with aggressive=true. The pass-level
     unsoundness remains OPEN; the misrouting is fixed.
#16  The reassociation path corrupts these wrapped-CT DAGs even at
     N=5 (VFFT_FORCE_REASSOC=1 -> 1.3e-01). Default-off, latent,
     needs its own hunt.
#17  Pre-registered symmetry-CSE bet (flops ratio <= 0.65, i.e.
     >=70% of the 2x saving recovered) PARTIALLY FALSIFIED:
     dct1_17/rdft_32 = 0.636 (72.8% recovered, WIN);
     dct1_33/rdft_64 = 0.685 (63.0%, LOSE);
     dst1_31/rdft_64 = 0.729 (54.2%, LOSE).
     Recovery degrades with depth: the mirrored subexpressions get
     buried behind twiddle cmuls that hash-consing cannot see
     through. Still: a generated DCT-I at ~2/3 the cost of the
     embedding rdft, against FFTW's second-class REDFT00.

INSTRUMENTS KEPT: bin/dbg_eval.exe (per-pass numeric evaluation,
permanent); VFFT_NO_SUBDEDUP and VFFT_FORCE_REASSOC env gates
(debug-only, documented here).

PENDING from this arc: REDFT00/RODFT00 rows in bench_trig_vs_fftw;
dct1/dst1 production shells (Phase 1, core/dct1.h pad-embedding for
arbitrary N); odd-N r2c Phase 1; scalar trig coverage.

## 57. Odd-N r2c/c2r Phase 1: embedding shell, gated, and a
## falsified deficit prediction

IMPLEMENTATION (core/r2c.h): stride_r2c_plan now dispatches odd N to
_r2c_plan_odd. Forward: full N-point complex FFT on (x, 0), serial,
natural-order half written through scratch un-permute (H = N/2+1
bins; odd N has no Nyquist). Backward: conjugate-forward identity
IDFT(X) = conj(DFT(conj(X))) through the SAME forward executor — no
dependence on the backward executor's ordering conventions, purely
real result for Hermitian input by construction. Reuses
stride_r2c_data_t so both shared convenience wrappers
(stride_execute_r2c / stride_execute_c2r) work verbatim; the
c2r_im_buf is sized N*K on the odd path as the Hermitian-fill
workspace. Inner plan contract: even N takes half-N (unchanged), odd
N takes full-N. Scaling matches even: c2r(r2c(x)) = N*x.

GATE (benchmarks/gate_r2c_odd.c, ALL PASS at 1e-11 threshold):
  N=8 even control 1.2e-15 / 3.5e-16 (convention witness);
  composites 9/15/21/105: 4.4e-15 .. 1.6e-13 fwd, all round trips
  clean; prime 17 planned directly by the auto planner; prime 31
  via Rader 7.7e-14; prime 47 via Bluestein 1.2e-13. Prime real N
  fell out free of the existing prime machinery, as predicted.
  One gate-authoring bug owned: the even control was first handed a
  full-N inner (even path requires half-N) — produced wrong values
  AND heap corruption from scratch overflow. Gate fixed; the
  production even path was never at fault.

FALSIFICATION LEDGER:
#18  Pre-registered Phase-1 deficit (FFTW faster 1.5-4x at odd N)
     FALSIFIED on both sides: N=15 K=256 vfft/fftw = 1.23 (below
     the band's floor); N=105 K=256 vfft/fftw = 0.64 — the naive
     embedding shell is 1.56x FASTER than FFTW PATIENT. Mechanism:
     FFTW's odd-N real path does not vectorize across the batch;
     our embedding rides the batched split engine at full width,
     repaying the 2x arithmetic. At N=15 the three N*K copy passes
     (memset im, un-permute staging, copy-back) dominate the tiny
     transform. Caveats: one machine (container Cascade Lake), two
     sizes, plan-bench noise ±4-5%. CONSEQUENCE: Phase 2 (optimal
     odd real-split algorithms) demoted — the shell already
     competes; revisit only if an odd-N user appears at small N
     where the copy overhead bites.

## 58. Boundary-kind production shells + the full nine-kind r2r sweep

SHELLS (core/dct1.h): stride_dct1_plan / stride_dst1_plan, house
contract (caller supplies the M-point r2c plan, ownership chains;
M = 2(N-1) resp. 2(N+1), always even, so the inner rides the fast
half-M path; plan rejected on M mismatch). Execute = extend into
buf_re, M-point r2c in place, extract (Re rows 0..N-1 for DCT-I —
M/2+1 = N bins exactly; -Im rows 1..N for DST-I). Both kinds are
involutions, so override_bwd = override_fwd. Serial Phase 1; MT
K-split and codelet dispatch at the gated sizes are the planner
integration follow-ups. 3-pointer convenience wrappers mirror
stride_execute_dct2.

GATE (benchmarks/gate_dct1_shell.c): ALL PASS, ten cells —
dct1 N=6/12/24/33/100 and dst1 N=4/10/22/31/99, inner M/2 spanning
smooth, prime-23-via-Rader, and 99/100 composites; fwd vs brute
1.5e-14..4.4e-13, involution scales clean. Arbitrary-N DCT-I/DST-I
is now a product fact, primes included.

BENCH (benchmarks/bench_trig_vs_fftw.c, restructured to a 36-cell
row table with per-kind sizes): the full nine-kind r2r family vs
FFTW PATIENT, home + split layouts, K=256, avx512, all 36
cross-checks PASS. New rows, home-layout speedups:
  dst2: 16.9 / 7.3 / 4.5 / 3.1x   (N = 8/16/32/64)
  dst3: 15.6 / 5.7 / 5.0 / 3.6x
  dst4: 17.8 / 6.0 / 5.4 / 4.0x
  dct1: 45.6 / 14.5 / 10.2 / 6.3x (N = 5/9/17/33)
  dst1: 71.2 / 30.9 / 11.3 / 5.4x (N = 3/7/15/31)
Split-layout column higher throughout. The dct1/dst1 cross-checks
are BITWISE exact (0.0e+00): FFTW's e00 path is evidently the same
pad-embedding algebra on a non-batched engine.

FALSIFICATION LEDGER:
#19  Pre-registered boundary-kind bands (dct1/dst1 home 2.5-12x,
     dst2/dst3 3-8x) FALSIFIED ABOVE at small N: dct1 N=5 hits
     45.6x, dst1 N=3 hits 71.2x, dst2/dst3 N=8 hit 16.9/15.6x.
     Within band at the larger sizes; dst4 (3-20x) landed entirely
     inside. The miss direction: I underestimated how badly FFTW's
     REDFT00/RODFT00 and small-N RODFT paths degrade under
     batching (N=5 REDFT00: 4886 ns vs our 107 ns). One machine,
     plan-bench noise ±4-5%, as always.

TIER K STATUS: COMPLETE. DST-IV, DCT-I, DST-I (codelets AND
arbitrary-N shells), odd-N r2c/c2r — every kind on the missing
list now exists, is gated, and beats FFTW PATIENT in every measured
cell. The "only r2r kinds where FFTW beats us by existing" sentence
is dead.

## 59. Three-way headline: vfft vs FFTW PATIENT vs MKL 2026

benchmarks/bench_headline_3way.c (container Cascade Lake, 1 vCPU,
K=256, single thread, all home layouts, mkl_rt). Harness bug owned:
the first c2c xcheck compared our digit-reversed raw stride output
against MKL's natural order — fixed perm-aware (7.1e-15 PASS);
r2c xcheck 3.0e-14 PASS.

  kind   N     vfft_ns   fftw/v   mkl/v
  c2c    64      19440    1.34x   0.63x
  c2c    256    128809    1.18x   0.86x
  c2c    1024  1003109    0.76x   0.63x
  prime  127    231958    2.45x   1.58x
  prime  257    709298    1.90x   1.41x
  r2c    64      20895    1.11x   0.79x
  r2c    256    146788    0.71x   0.49x

STALE-WISDOM PROBE: in-container exhaustive-patient at N=1024 finds
(4,16,16) at 847614 ns — 1.18x over the repo-wisdom plan. So 18% of
the 1024 deficit is wisdom staleness alone (the standing top-priority
wisdom-regen item, now with in-container proof); the remainder is
where A9 (pow2-stride aliasing, +16% measured) and large-N traffic
live. Even exhaustive-best trails MKL at 1024 (0.74x): commodity
large pow2 is MKL's home turf on this box. r2c 256 inherits the same
pow2 inner and adds pre/post passes — same medicine applies.

MKL n/a for the r2r family (no batched DCT/DST; TT is
single-sequence legacy) — those rows remain the FFTW-only 36-cell
sweep of section 58.

FALSIFICATION LEDGER:
#20  Pre-registrations scored: pow2 ±30% band held at 64/256
     (0.63x at 64 vs MKL is below band — MKL stronger at small pow2
     than predicted) and BROKEN at 1024 vs MKL (0.63x); the N=64
     possible-win called half right (won FFTW, lost MKL). Prime band
     2.5-3.2x vs FFTW came in at 2.45/1.90 — BELOW the prior
     session's 2.66-3.12 measurements (different harness fill +
     day-to-day container variance; direction unchanged, magnitude
     not replicated — flagged for re-measurement on real hardware).
     Prime-vs-MKL 1.41-1.58x landed inside the wide 0.7-3x band.

PRIORITY REINFORCEMENT: the two cells we lose are exactly the two
open items' home (i9/per-box wisdom regen, A9). The cells we own
(primes, all nine r2r kinds, batched split) are untouched by either
library.

## 59b. Exhaustive-patient re-plan of the headline cells

User verdict to test: "those factorizations are really bad." Searched
every vfft cell (incl. Rader and r2c inners) with exhaustive-patient
on this box, then re-benched 3-way with the found shapes.

FOUND vs WISDOM:
  32  -> (4,8)      = wisdom (unchanged)
  64  -> (8,8)      vs wisdom (4,4,4)
  128 -> (4,4,8)    = wisdom (unchanged)
  256 -> (2,4,4,8)  vs wisdom 4^4
  1024-> (4,16,16)  vs wisdom 4^5
  126 -> (6,7,3)    vs wisdom (6,3,7) (same multiset, reordered)

RE-BENCH (exhaustive shapes, fresh FFTW PATIENT + MKL columns):
  c2c 1024 (4,16,16): 819561 ns -> 0.93x FFTW (was 0.76x),
    0.76x MKL (was 0.63x). The -18% is real and reproduces.
  c2c 256 (2,4,4,8): 125821 ns, -2% = noise. Tie with 4^4.
  primes / r2c: within noise of the wisdom plans; the (6,7,3)
    reorder is a wash; r2c inners were already optimal.

N=64 ANOMALY RESOLVED: phase-C measured (8,8) at 23116 ns vs the
wisdom plan's 19440 — but a controlled duel (same constructor, three
timing disciplines) shows the shapes TIED: (4,4,4) vs (8,8) =
19444/20271 hot, 20248/20855 busted-trial, 44070/42665 per-call-cold.
The 23116 was run-to-run spread plus default-variant construction
(explicit plan_create with variants=NULL vs the wisdom plan's
recorded variant codes), not the factorization. Separately: the
exhaustive search's internal estimate (12207 ns for the same plan) is
hot-tight-loop methodology, ~1.6x optimistic in absolute ns at this
size — search estimates rank candidates but must never be compared
against harness numbers. WISDOM-REGEN NOTE (for the i9 job): the
regen's measurement discipline must match deployment conditions;
on this box the discipline did not flip rankings at N=64, but the
absolute scales differ enough to mislead if mixed.

VERDICT ON THE WISDOM FILE (this box): exactly ONE genuinely bad
entry — 1024 K=256 (4^5, worth -18%). Everything else tied or
already optimal. CORRECTED HEADLINE TABLE (best measurements):
  c2c    64    1.3x FFTW   0.64x MKL
  c2c    256   1.3x FFTW   0.86x MKL
  c2c    1024  0.93x FFTW  0.76x MKL   (was 0.76/0.63 pre-fix)
  prime  127   2.2-2.5x    1.5-1.6x
  prime  257   1.9x        1.4x
  r2c    64    1.1x        0.78x
  r2c    256   0.7x        0.47x       (structural: inners optimal)
The remaining pow2/r2c gap to MKL is now cleanly attributed:
A9 pow2-stride aliasing (+16% measured, unimplemented) and the r2c
pre/post layer vs MKL's tuned real path — not planning.

PRE-REGISTRATION SCORE (all in band this time): 1024 ~18% (exact);
64/256 0-15% (tie and +2%); primes 0-15% (tie); r2c 0-20% (0);
still-trailing-MKL-at-1024 0.70-0.85x (0.76x).

## 59c. r2c/c2r codelet types and the r2c-256 decomposition

CODELET TYPES (r2c fwd): the inner half-N complex transform uses the
SAME generated DFT family as c2c — stage 0 is a no-twiddle n1_fwd
leaf (radix{R}_n1_fwd_avx512) called with a real-input even/odd leg
pattern (the half-complex pack fused into the leaf via pointer
arithmetic, _r2c_fused_first_stage), and stages 1..n are t1 twiddle
DIT/DIF codelets (radix{R}_t1_dif_fwd_avx512). Then _r2c_postprocess
is a HAND-WRITTEN avx512/avx2 Hermitian recombination (DC/Nyquist
special-cased, general twiddle butterfly over N/2 factors, with a
digit-reversal perm-gather on the inner output) — not a generated
codelet. c2r is the mirror: hand-written inverse-recombination
preprocess + inner backward (t1_*_bwd) + fused n1_bwd last stage.

DECOMPOSITION of r2c-256 K=256 (the surprising MKL loss):
  inner c2c-128 (4,4,8) alone : 35223 ns   <- HALF of MKL's whole 71469
  r2c-256 in-place (override) : 76813 ns   (+41591 recombination)
  r2c-256 via 3-ptr API       : 114901 ns  (+38088 API input memcpy)
  MKL r2c-256: 71469   FFTW: 102246

FINDING: our codelets are not the problem — the inner FFT alone beats
MKL's entire real transform. The deficit is 100% r2c GLUE:
  (1) the 3-pointer convenience memcpy (+38us) — avoidable by
      zero-copy in-place use; the in-place path (76.8us) is already
      ~1.07x of MKL (near parity).
  (2) the postprocess recombination (+42us, > the FFT itself) — the
      perm-gather (scattered loads via digit-reversal) is the cost;
      MKL's fused real kernel writes natural order with no separate
      gather pass.

ACTIONABLE (new tier-A candidates):
  A11  Fuse _r2c_postprocess into the inner's LAST stage — the dual
       of the already-shipped fused-first-stage on the forward pack.
       Kills the separate O(N*K) gather pass; targets the +42us.
       Alternatively route the inner through a natural-order-output
       (DIF-last) plan so the perm-gather vanishes.
  A12  Document/encourage the zero-copy in-place r2c entry; the
       3-ptr wrapper's memcpy is half the MKL gap at N=256.
These were invisible until the decomposition; they likely apply to
the c2r preprocess and to the dct/dst shells (which call r2c) too.

## 59d. A12 shipped: out-of-place r2c forward (kill the API memcpy)

After 59c falsified A11 as the big lever (mirror-scatter only +2.7us,
14% of postprocess; the recombination + strided pack are ~inherent to
the half-complex method and already ~1.07x MKL in-place), the real
win was the 3-pointer convenience memcpy (~31-38us at N=256 K=256).

FIX (additive, core/r2c.h): _r2c_execute_fwd_oop + _r2c_worker_fwd_oop
read `in` directly and write (out_re, out_im), reusing the SAME
_r2c_fused_first_stage and _r2c_postprocess helpers (strictly less
aliasing than in-place). stride_execute_r2c now routes even-N plans
to the OOP path (no pre-copy); odd-N (section 57) keeps the copy
route. The in-place override (_r2c_execute_fwd) is untouched — the
dct/dst shells and any zero-copy caller are unaffected.

CORRECTNESS (benchmarks/gate_r2c_oop.c): OOP output BIT-IDENTICAL
(0.0e+00) to the old copy-then-in-place path at N=64 and 256, and
matches brute Hermitian (8.1e-14 / 6.6e-13). Odd-N gate still ALL
PASS (shared wrapper change is parity-safe).

PERF (r2c-256 K=256, controlled same-environment delta):
  old 3-ptr API (memcpy + in-place): ~114900 ns
  new 3-ptr API (OOP, this fix):     ~83900 ns   -> 27% faster
  pure in-place (clobbers input):    ~76800 ns   (fastest; via
                                       stride_execute_fwd)
The 31us memcpy is eliminated; OOP costs ~7us more than pure in-place
because it writes separate output buffers (preserving the input),
the correct tradeoff for the general API.

FALSIFICATION LEDGER:
#21  Pre-registered OOP r2c-256 ~75-80 ns-thousand and mkl/v
     0.92-1.10x. OOP came in at 83.9 (above band: separate-output
     traffic vs pure in-place) and mkl/v 0.66x THIS run because MKL
     measured 55.7us (vs 71.5us last run — container MKL variance
     is ~+-25%). The robust, controlled claim is the same-environment
     API delta (115->84us, -27%); the absolute MKL ratio is noisy
     and we remain behind MKL's fused real kernel (0.66-0.85x across
     its timing spread). A11 (recombination fusion) stays unbuilt by
     design — its ceiling is ~3us.

FOLLOW-UP: c2r has the identical memcpy in stride_execute_c2r; the
symmetric OOP backward is the same additive pattern (deferred — the
forward was the measured cell).

## 59e. CORRECTION: the r2c-256 deficit is real-glue, not a "fused kernel"

Sections 59c/59d asserted MKL has a "tight/fused real kernel" — that
was an unverified just-so story (inferred from "MKL is faster" + a
prior). WITHDRAWN. Component measurement, same environment, MKL
confirmed single-threaded (mkl_get_max_threads()=1 on a 1-CPU box,
so no threading confound):

  our c2c-128 (4,4,8) : 35888 ns
  MKL c2c-128         : 37305 ns   (1.04x — TIED; we are marginally
                                    faster, MKL's base FFT is NOT
                                    the advantage)
  MKL r2c-256         : 56516 ns
  our r2c-256 in-place: ~76800 ns

Real-transform overhead, relative to each library's own c2c-128:
  MKL:  r2c-256 / c2c-128 = 1.51  -> real overhead ~0.51x a c2c-128
  ours: r2c-256 / c2c-128 = 2.13  -> real overhead ~1.13x a c2c-128
Our half-complex recombination+pack (~41us) is roughly 2x MKL's
real overhead (~19us).

GROUNDED CLAIM: the r2c-256 gap is NOT kernel quality (complex FFTs
tied); it is the real-transform-specific glue, where our overhead is
~2x MKL's. The MECHANISM of MKL's lower overhead is UNKNOWN — could
be a native real-FFT algorithm, a cheaper recombination, or fusion;
not determined (no MKL profiling available here). Earlier "fused
kernel" wording in 59c/59d should be read as this unverified guess.

Related softening: the c2c-1024 gap was attributed to A9
(pow2-stride aliasing). A9 is a PRIOR-measured +16% pow2 item, but it
has not been re-verified as THE cause of this specific 1024 gap this
session — treat that attribution as informed-but-unconfirmed too.

## 59f. Where the r2c real overhead is, and how to reduce it (measured)

The ~41us real overhead (r2c-256 in-place ~77us minus inner c2c-128
~36us) decomposes, measured same-environment:

  pack / fused-first strided phase : ~21 us  (Path A fused 70us vs
       Path B explicit-pack 68us = TIE; pack alone 21.5us. Fusion
       vs separate pack is noise — the cost is moving the data.)
  postprocess phase                : ~28 us  (memcpy same volume
       21.6us = TRAFFIC floor; only +6us compute/gather. 78%
       traffic-bound. mirror-scatter is 2.7us of this, sec 59c.)

DIAGNOSIS: both halves are TRAFFIC, not compute. The half-complex
method makes two extra full-array passes vs a native real FFT:
pack (read in, write scratch) and postprocess (read scratch, write
out). MKL's ~19us real overhead (sec 59e) is ~one fewer pass — by
a native-real strategy we have not determined.

REDUCTION MENU (measured ceilings, not guesses):
  1. Fuse postprocess into the inner's LAST stage. Removes ~the
     scratch-READ half of the postprocess traffic. MEASURED ceiling
     ~10us (half of the 21.6us pp traffic floor) -> r2c-256 in-place
     ~77 -> ~67us. Additive, medium effort/risk. NOT about the
     scatter (2.7us) — about eliminating a memory pass. This is the
     reframed A11.
  2. Pack (~21us): traffic-bound moving ~1MB; the fused-first path
     already reads `in` once. Reducing it needs cross-stage register
     fusion (keep stage-0 output in regs for stage 1) — large, and
     the inner is a generic mixed-radix executor, not fusion-friendly.
  3. FUNDAMENTAL FIX (matches MKL): a native real mixed-radix FFT
     using the generator's existing rdft (real-input DFT) codelets,
     handling conjugate symmetry throughout — no bulk pack, no
     separate recombination, so neither extra pass exists. Big
     project: the plan/executor is complex-only today; rdft codelets
     exist but only monolithic, so a real mixed-radix planner is the
     work. This is the only path to MKL-parity on r2c.
  4. (Niche) two-reals-in-one-complex for paired real-signal
     workloads — halves per-signal cost; only helps that use case.

HONEST SUMMARY: #1 is the tractable ~10us win; #3 is the real answer
but a major effort. The pack is near a traffic floor. None of the
overhead is kernel quality — our complex FFT is tied with MKL (59e).

## 60. Native rfft P0: the hc2hc/hc2c port validated

Design: docs/native_rfft_design.md. P0 de-risk, pre-registered
"composition gate passes at 1e-13 within <= 2 fix iterations".
Result: ZERO fix iterations.

Discovered state: the generator port is more complete than the
design assumed — --hc2hc (DIT and DIF) AND --hc2c (the cascade
terminator, i.e. the design's HC2HC_LAST) are fully wired with
documented contracts:
  hc2hc DIT: out = sym1(sym2(DFT_n(byw(in))))
  hc2hc DIF: out = byw_post(DFT_n(sym2i(sym1(in))))
  hc2c  DIT: out = sym(DFT_n(byw(in)))
  sym1: re passthrough, im[i] <- im[n-1-i]
  sym2: upper half (2i >= n) rotate +i;  sym2i: rotate -i
  sym : upper half conjugate
ABI: generic 7-arg, rows [slot*K + lane]; twiddle slot 0 never
loaded (proved by NaN-poisoning row 0 in the gate — any leak would
NaN the outputs).

GATES (benchmarks/gate_hc_codelets.c, K=8, all rows x lanes vs the
contract formulas in C):
  r=4: hc2hc_dit 1.1e-15, hc2hc_dif 1.1e-15, hc2c_dit 1.1e-15  PASS
  r=8: hc2hc_dit 1.4e-14, hc2hc_dif 1.2e-14, hc2c_dit 1.4e-14  PASS
r=8 exercises the CT(2,4) internal path.

POLICY KEYING (section-56 lesson): hc2hc/hc2c flag n equals the
internal Dft.dft n, so aggressive/reassoc/fma keying is correct BY
CONSTRUCTION (unlike dct1/dst1). At r=4/8: CT -> aggressive=false.
Odd radices (3/5/7) will key Direct -> aggressive=true over
cmul+conjugate-pair DAGs — the validated t1-prime shape plus sym
index shuffles; to be explicitly gated in P1, not assumed.

P0 verdict: the math layer, simplification pipeline, emit, and ABI
all hold for the real-cascade codelets. Risk #1 of the design is
retired. D2 (HC2HC_LAST) is cheaper than designed: hc2c exists; the
remaining D2 work is only the natural-SPLIT-store emission variant
(hc2c currently emits packed-position Output refs; the executor or
an emit variant maps them to (out_re,out_im) natural rows).

P1 next: r2cf leaf ABI variant (rdft math on n1 leg-stride
plumbing), odd-radix hc2hc gates, coverage quadrant rfft-{isa},
per-codelet gates, regen.

## 61. Native rfft P1 complete: r2cf leaf, odd-radix clearance, rfft quadrant

Design: docs/native_rfft_design.md. P1 pre-registrations: (i) odd-radix
hc2hc (aggressive=true policy path) gates green WITHOUT pipeline fixes;
(ii) r2cf rides a new emit mode with <= 1 emitter fix (asymmetric
output-set risk). Scored: (i) TRUE; (ii) ZERO fixes.

ABI ARCHAEOLOGY (the decision that shaped r2cf): the executor's leaf
typedef stride_n1_fn (core/stride_executor.h:254) is 7-arg
  (const double* in_re, const double* in_im,
   double* out_re, double* out_im, size_t is, size_t os, size_t vl)
and NO existing generator quadrant emits that shape — in-place n1 is
6-arg (rio/tw/ios/me), oop n1 is 11-arg (group strides + UG modes),
strided n1 is 6-arg in-place. So r2cf emits the executor typedef
DIRECTLY (zero shims for P2); how n1_fwd_table gets filled today
(stride_executor.h:1783/1799) is P2 registry archaeology.

GENERATOR DELIVERABLES:
- lib/dft_r2c.ml dft_expand_r2cf: the rdft minus its identically-zero
  imaginary outputs — re k=0..n/2, im k=1..im_hi where
  im_hi = (n even ? n/2-1 : n/2). Both parities emit exactly n outputs:
  the constant-footprint halfcomplex leaf (design D1/D3).
- lib/emit_c.ml r2cf_signature mode (4 patch points): typedef-exact
  signature with (void)in_im, in_re[j*is+v] / out_*[k*os+v] addressing,
  vl loop, hoisted consts.
- lib/gen_main.ml --r2cf flag; naming radix{r}_r2cf_{isa}; r2cf added
  to the hoist disjunction. Policy keying correct BY CONSTRUCTION
  (flag n == internal rdft n).

GATES (avx512, K=8, vs real-DFT formula, skipped-row sentinels 7777):
  r2cf r=4 4.4e-16 | r=8 8.0e-15 | r=5 1.2e-14 | r=2 0.0 | r=16 1.5e-14
  ALL PASS, sentinels untouched at im row 0 (and n/2 for even n).
ODD-RADIX hc FAMILY (the flagged aggressive=true risk, now retired):
  r=3: dit/dif/hc2c 3.1e-15 | r=5: 1.7-1.9e-14 | r=7: 1.9-2.2e-14  PASS
BOUNDARY+PLAN RADICES: r=2 hc trio 2.2-4.4e-16 | r=16 hc trio
  3.7-7.3e-14  PASS.
AVX2 RENDER PATH (previously ungated ISA): hc trio r=4 at 1.1e-15 and
  r2cf all five radices identical-to-avx512 error class, sentinels
  PASS — gated against the TREE artifacts, not /tmp builds.

COVERAGE QUADRANT rfft-{avx2,avx512} (lib/coverage.ml): radices
{2,3,4,5,7,8,16} x {r2cf, hc2hc dit fwd, hc2hc dif fwd, hc2c dit fwd}
= 28 files/ISA, 56 total. Forward only in P1 (backward lands with
c2r). Naming agreement spot-verified; 56/56 compile (avx512 and avx2).

FLAG-LEAKAGE PROOF, with an owned correction: the first "byte-identical"
check was a NO-OP (scripts/regen_codelets.sh does not exist in this
tree; nothing had regenerated). The real proof: gen_set.exe over all
ten quadrants regenerated the full tree, after which all 825
pre-existing files are byte-identical (md5 manifest diff). Tree:
825 -> 881.

Gate sources promoted: benchmarks/gate_r2cf.c (avx512 canonical;
avx2 variant is the same source with avx512->avx2 substitution).
Object caches for P2: /tmp/rfft_o512, /tmp/rfft_o2 (all 56 .o).

P1 VERDICT: every codelet the native rfft executor needs exists in
the tree, gated on both ISAs, with the leaf already speaking the
executor's exact ABI. P2 is now pure runtime work: stage-kind enum
{N1,T1,R2CF,HC2HC,HC2HC_LAST}, registry archaeology for table fill,
rfft plan builder on the existing factorizer, wisdom key "r2hc:N:K",
the D2 residual (hc2c natural-split-store: emit variant vs executor
mapping), and cascade composition gates (children r2cf + hc2hc ==
monolithic rdft) at N=16..1024.

## 62. P2 part 1: composition algebra, ABI v2, first cascades compose

### The composition contract (derived, then verified empirically)
For parent size n_p = r*m, column k in (0, m/2): pre-twiddle the r
child values Z_j[k] by W_{n_p}^{jk} and take an r-point DFT across j
— exactly the hc2hc formula byw + DFT_r. Checking ALL EIGHT output
slots at r=4 symbolically against FFTW packed positions: sym1∘sym2
lands every value at the packed parent slot (re-stream base k stride
m; im-stream base m-k stride m, both POSITIVE — sym1's index reversal
is precisely what absorbs the mirror). Corollaries:
  - the k=0 column IS an r2cf call (Z_j[0] real, W^0=1): no new codelet;
  - the k=m/2 column (m even) is self-mirror: handled by a small
    executor-direct loop (one column per stage, negligible);
  - the LEAF's packed im slots walk positions rho-t, i.e. BACKWARDS.

### ABI v2 (P1 revision, ledger entry)
The leaf reversal falsifies P1's ABI choice: "r2cf speaks stride_n1_fn
exactly, zero shims" is WITHDRAWN — composition needs signed, split
output strides. v2 ABIs (regenerated, quadrant only):
  r2cf:  (const double* in_re, double* out_re, double* out_im,
          ptrdiff_t is, ptrdiff_t os_re, ptrdiff_t os_im, size_t vl)
         executor passes os_im < 0 with out_im based one-past.
  hc2hc/hc2c strided: (in_re, in_im, out_re, out_im, tw_re, tw_im,
          ptrdiff_t is, ptrdiff_t os, size_t vl); twiddles replicate
          per vl lanes (tw[j*vl + v]), slot 0 never loaded.
Rationale recorded: the generic ABI's hardcoded slot stride K cannot
address middle stages (strides are Q*K multiples, in != out stride).

### Buffer choreography v1: ping-pong
Per-stage ping-pong between two N x K planes (in-place collision
analysis deferred). Footprint = the complex executor's re+im pair, so
memory parity holds; the native path's win is eliminating the pack
and postprocess PASSES, which ping-pong preserves.

### Mid-stage batching (derived, NOT yet tested)
For middle stages (Q = product of radices above > 1), subproblem
index g and lane index fold into one contiguous vector dim:
vl = Q*K, slot strides in units of Q*K. This collapses Q subproblem
calls into one codelet call per column. UNTESTED — the three cells
below are all two-stage (Q=1); the multi-stage harness is the next
gate before any executor code.

### Evidence this section
- Flag leakage: 825 non-rfft files BYTE-IDENTICAL across two
  quadrant regens. 56/56 compile both ISAs.
- Codelet re-gates under v2: r2cf {2,4,5,8,16} 0..1.5e-14 +
  sentinels; hc2hc dit/dif + hc2c at {2,3,4,5,7,8,16} 2.2e-16..
  7.3e-14 — identical error class to v1 (ABI change is pure plumbing).
- COMPOSITION (benchmarks/gate_rfft_compose.c, leaf instrument +
  total vs brute packed): N=16 (4,4) 6.8e-15; N=8 (2,4) 1.1e-15;
  N=8 (4,2) 1.1e-15 — ALL PASS, ZERO wiring-fix iterations.

### Pre-registration scoring (this turn)
- "regen clean + gates re-pass": MISSED by one emitter fix — the
  generic twiddle render hardcoded lane var `k`; strided hc loops use
  `v`. Fixed by making the Twiddle render follow loop_var.
- "composition within <= 3 wiring-fix iterations": BEATEN — zero.

NEXT: generalize the harness to L-stage plans with the full odometer
(tests the Q>1 batching), cells N=32(2,4,4), 64(4,4,4)/(2,4,8),
128(2,4,4,4), odd-mix 20(5,4)/24(3,8); then transcribe the PROVEN
loop into core/rfft.h (stage-kind enum, plan builder, wisdom key),
then hc2c last stage + D2 natural-split store.

## 63. P2 part 2: L-stage proven, core/rfft.h shipped, smoke bench misses

### L-stage harness (benchmarks/gate_rfft_compose_L.c)
The generalized reference loop with the Q-fold (vl = Q*K) and the
per-depth instrument. TWELVE cells, ALL PASS, ZERO fix iterations
(pre-reg <= 2): (4,4) | (2,4,4) first Q>1, instrument clean at every
boundary | (4,4,4) | (2,4,8) | (2,4,4,4) Q to 8 | (4,4,16) and
(2,4,4,8) the 256 target shapes, 8.8e-13 | (5,4) (3,8) odd combine |
(4,5) ODD m, no Nyquist column | (2,3,2) m=2 stage, no interior |
(7,3,5) all-odd N=105. The Q>1 batching trick is now empirical fact.

### core/rfft.h (the transcription)
rfft_plan_create(N, K, factors, nf, registry) with factors[0] =
outermost combine, factors[nf-1] = leaf; precomputed vl-replicated
twiddles (packed from k=1) and mid-column coefficient tables;
ping-pong planes (one scratch for nf<=2, two for nf>=3); the d=0
stage writes the caller's output buffer directly; nf=1 pure-leaf
edge supported. K%8==0 required (vl vector-width multiple). Output
v1 = PACKED halfcomplex N x K plane (hc2c natural-split replaces the
d=0 stage in the next phase).
GATE (benchmarks/gate_rfft_plan.c): 12 plan-driven cells incl K=64
variation and nf=1 — ALL PASS, ZERO transcription fixes (pre-reg
<= 2); K=8 errors BIT-IDENTICAL to the harness (faithful transcription).

### Smoke bench: pre-registration MISSED, honestly
Pre-registered smoke band 45-65us for rfft-256 K=256 packed. Measured:
  (4,4,16):  161025 ns     (2,4,4,8): 226081 ns
2.5-3.5x ABOVE band; 2-3x slower than the half-complex wrapper
(76.8us) it is meant to replace. The two points are ~linear in stage
count (~54-57us/stage), which fingers streaming-bound full-plane
stages: each stage moves read(512KB) + write(512KB) + replicated
twiddles(~512KB) ~ 1.5MB, spilling L2 — while the half-complex path
keeps its inner c2c-128 (512KB) L2-resident and streams only twice.
DIAGNOSIS IS SUSPICION, not yet proof.

Ranked fixes (next turn): (1) scalar-broadcast twiddles for hc2hc —
the t1s machinery already exists in the emitter; removes ~1/3 of
per-stage traffic and most twiddle table memory; (2) in-place or
K-blocked execution to cut ping-pong plane traffic (the complex
executor's B-blocking is the model); (3) column scheduling for write
locality (out slots are m*vl apart). The P3 target (<= 62us) needs
~2.7x per-stage improvement: plausible from (1)+(2), NOT a foregone
conclusion. Correctness architecture is DONE; the contest is now
purely bandwidth.

### Pre-registration scoring (this turn)
- L-stage harness <= 2 fixes: BEATEN (zero).
- rfft.h transcription <= 2 fixes: BEATEN (zero).
- Smoke band 45-65us: MISSED 2.5-3.5x (ledger #22). The miss is the
  most informative datum of the turn: codelets and geometry are
  right; the v1 EXECUTION SCHEDULE is bandwidth-naive.

## 64. "How can fewer passes lose?" — answered by decomposition

The question (user's, exactly right to ask): the native path baked the
pack and recombination into the cascade; how can it be slower than the
wrapper that pays them?

### The accounting that reframes it
Raw traffic at N=256 K=256: half-complex = pack(~1MB) + 3 in-place
inner stages(~3MB) + postprocess(~1MB) ~ 5MB. Native (4,4,16) =
leaf(1MB) + 2 ping-pong stages(~2MB) + twiddles(~0.6MB) ~ 3.6MB.
THE NATIVE PATH MOVES LESS. The "fewer passes" premise held at the
volume level; the loss was in execution, not architecture.

### Falsification ledger #23: this turn's predictions
Pre-registered: (a) leaf >= 40% of runtime (same-set 32-stream
thrash); (b) (16,4,4) <= 0.7x (4,4,16). MEASURED: leaf = 17% at
34GB/s — the supposed aliasing monster is the FASTEST phase per
byte; (16,4,4) = 1.9x WORSE. Both falsified. Section 63's L2-spill
story and this turn's set-aliasing story were both wrong as primary
mechanisms.

### The real elephant (found by per-phase + factor-order sweep)
The k=m/2 "negligible" direct column (section 62 claim: FALSIFIED on
the ledger) is O(Q*K*r^2) SCALAR work, and Q multiplies the lanes.
At (4,16,4): the r=16, Q=4 mid is ~4096 lanes x 512 ops ~ the entire
472us anomaly. The sweep ordering correlates with sum of Q*K*r^2
across stages, not with stream-aliasing.

### Fix shipped: vectorized mid column (core/rfft.h)
AVX-512/AVX2 broadcast-FMA over the coefficient table on the folded
v dim (mid inputs are contiguous in v exactly like codelet streams),
scalar tail fallback. All 12 plan gates PASS unchanged (5.64e-13 at
(4,4,16) K=64, identical pre/post).

RESULTS (pre-registered bands BOTH HIT):
  (4,4,16): 170 -> 146us  [band 125-150 ✓]
  (4,16,4): 472 -> 195us  [band 150-250 ✓]
  best plan now (16,16): 103us
  (others: (4,8,8) 151, (8,4,8) 152, (16,4,4) 161, (2,4,4,8) 174)
NOTE: bench_rfft_phase.c's bench-local run_stage retains the scalar
mid (its per-stage splits are pre-fix); the sweep numbers come
through rfft.h and are authoritative.

### Standing
(16,16) 103us vs half-complex 77us vs MKL ~56 (noisy): the native
path is now 1.34x behind its replacement target with two ranked
levers unspent:
  1. t1s-style BROADCAST twiddles for hc2hc (emitter machinery
     exists): at (16,16) the replicated tables are ~0.46MB of the
     single combine stage's ~1.5MB — both bandwidth and memory win.
  2. LANE-BLOCKING: lanes are independent, so running the whole
     cascade over K-chunks (Kb ~ 64) makes the ENTIRE transform
     L2-resident — the half-complex inner's secret, recreated.
     Caveat: Kb < K breaks the (q,lane) Q-fold, so blocked stages
     loop q explicitly (Q small calls of vl=Kb); modest restructure.
The "passes" premise is vindicated; the contest is schedule quality.

## 65. The two levers, scored: one masked, one NEGATIVE; codelets exonerated

Pre-registered: broadcast twiddles + lane-blocking land (16,16) in
[50,75]us and (4,4,16) in [65,100]us. MEASURED after both: 106 / 147
— BOTH MISSED with ~zero net movement. The decomposition that
followed is the content of this section.

### Lever scoring (falsification ledger #24)
1. BROADCAST TWIDDLES (--t1s for hc family): shipped; all 21 codelet
   gates green; 825 tree files byte-identical. Effect initially ~0 —
   because lever 2 was masking it.
2. LANE-BLOCKING: NEGATIVE. Kb=K (off) vs Kb=96: (4,4,16) 144 ->
   112us (-22%). The slab heuristic cut per-stream bursts to ~768B,
   defeating prefetch — and the cascade was never capacity-bound
   (that hypothesis died in section 64). Default now Kb = K; the
   mechanism remains a tunable.
3. SPILL-RECIPE EXTENSION for hc: NO-OP, reverted. should_spill(16)
   is true and the flag plumbs, but the spill machinery lives in the
   t1/n1 BUILDERS' blocked construction, not as a generic DAG pass —
   the hc builder has nothing to hook. radix16 hc stays an 82-spill
   storm (10% of insns are rsp traffic); radices <= 8 are clean
   (<= 8 spills).

### Codelets exonerated (bench_rfft_micro.c)
Hot-buffer throughput: r2cf_4 175 GB/s, hc2hc_4 140 GB/s, r2cf_16
54 GB/s, hc2hc_16 38 GB/s (the spill storm costs r=16 ~3.6x/byte vs
r=4). In-plane context strides cost only 1.4x on top. The 5-11x
in-plan gaps of section 64's phase data were the COMPOUND of:
replicated-twiddle traffic + blocking's short bursts + (at r=16)
spills — not any single villain, which is why single-lever
predictions kept missing.

### Standing after the chain (sweep, Kb=K, scalar tw, per-q)
  (16,16)   84.9us   <- best; was 103 at section 64, 161 at smoke
  (4,4,16) 116.3     (2,4,4,8) 127.6   (8,4,8) 129.8   (4,8,8) 132.0
vs half-complex in-place 76.8us, MKL ~56 (same-environment, noisy).
The native path is now 1.10x from its replacement target. All 12
plan gates PASS unchanged throughout.

### Next levers, ranked by measured headroom
1. hc2hc_16 spill fix (builder-side blocked construction for hc DAGs,
   or a structural 16 = 4x4 two-level hc decomposition): the (16,16)
   combine is ~55us of which ~24us is seven 3.4us spill-stormy
   columns that should cost ~1.4us each clean -> (16,16) ~ 60-65us,
   i.e. the ORIGINAL pre-registered acceptance (<= 62us) back in
   direct reach.
2. Leaf throughput: 30us for 1MB (33 GB/s) vs r2cf_4's 175 hot —
   the 16-stream 32KB-stride pattern leaves margin.
3. Post-fix per-phase decomposition FIRST next turn (the stale phase
   bench retired); no more single-cause stories without it.

## 66. Decomposition + the r16 fix attempt: shipped pieces, open item,
## and the wall renamed

### In-header profile instrument (VFFT_RFFT_PROFILE)
Per-phase accumulators (leaf / per-stage k0 / cols / mid) inside
rfft.h, compile-gated, zero production cost. Retires the stale-bench
failure mode permanently. benchmarks/bench_rfft_decomp.c.

### Decomposition (profile-on totals ~+9us)
  (16,16)  93.5: leaf 34.4 | k0 2.2 | cols 41.2 | mid 15.0
  (4,4,16) 119 : leaf 45.1 | d1 {k0 1.7, cols 33.1, mid 6.8}
                          | d0 {k0 0.6, cols 29.0, mid 1.5}
  (8,32)   91.3: leaf 43.9 | k0 0.8 | cols 40.0 | mid 6.1

### Mid column: s-blocked (shipped)
The unblocked vector mid kept 33 ZMM live -> gcc spilled accumulators
(section 66 opener). s-blocked (SB=4) shipped; r=4 mids improved
(9.0->6.8, 2.0->1.5) but the r=16 mid is RESISTANT (16.3->15.0):
it is a 16x16 per-lane GEMV — 512 broadcast+FMA per v-iter,
dependency/broadcast-port bound, ~0.4 IPC. Two named candidates,
unbuilt: (i) pre-broadcast coef table (16KB) turning broadcasts into
full-vector loads; (ii) route the mid through the hc2hc codelet with
a zero im-input (section 62 proved out_re alone is the correct packed
mid). Deferred: see the reframe below for why.

### r16 spill fix: direct attempt (partial), sidestep (gated), and
### a correction
- SHIPPED: dft_expand_hc2hc_spill / hc2c_spill — the t1 spill builder
  + output-side syms (DIT only; DIF pre-syms inputs and has no spill
  route; executor uses DIT only). Gen_main dispatch + recipe trigger.
- RESULT: 82 -> 79 rsp-vmovs. The spill MARKERS do not engage the
  emit-side bounding for the hc path — zero spill_pass/regalloc log
  lines even with explicit --spill. OPEN ITEM with probe trail.
- CORRECTION on the record: production t1p_16's zero-spill comes from
  --twiddled-pos (per-position twiddle policy/structure), NOT the
  spill recipe — my section-65 premise ("the recipe fixes this") was
  partly wrong. The per-pos policy is itself a candidate route for hc.
- SLOT UNIFICATION (load-bearing): plain hc builders moved to the
  TP_Flat convention (leg j -> Twiddle slot j-1) so plain and
  spill-built codelets agree across all radices and ISAs. Gate and
  executor fill updated together; ALL 27 codelet gates + ALL 14 plan
  gates PASS.
- SIDESTEP: radix-32 r2cf leaf (leaf-only coverage entry; 70 spills,
  4.2e-14 gate) enabling 256=(8,32) with no r16 combine: 91.3us
  profile-on — parity with (16,16), no breakthrough. Tree 825+58=883.

### THE REFRAME (what the three decompositions agree on)
cols cost 29-41us in EVERY plan regardless of radix mix, at
0.9-5.9us/call vs 0.23-3.4 hot — a 2-4x CONTEXT penalty on every
call, spill-free codelets included ((4,4,16) d0: r=4, zero spills,
still 4x). The wall is not r16, not spills, not twiddles, not
capacity: it is per-call latency exposure walking cold pow2-strided
rows. r16 spills and the r16 mid are real but SECOND-order behind it.
Next levers, reordered accordingly: (1) software prefetch of the next
column's rows inside the executor loops; (2) column scheduling for
line reuse (adjacent k share nothing, but k and its in/out row
neighborhoods interleave); (3) per-pos twiddle policy for hc (the
t1p_16 lesson). Standing best: (16,16) ~ 85us clean vs 76.8
half-complex, acceptance <= 62 still open, gap now correctly named.

## 66b. Which codelet-performance machinery reaches the hc/r2cf family
## (user question; two prior claims corrected)

The pipeline, with verified status for hc2hc/hc2c/r2cf:
  algsimp / CSE / sub-dedup / FMA fusion   YES (P0, dbg_eval lineage)
  constant hoisting                         YES (enabled section 61)
  SU scheduler                              YES (--su; pins present)
  emit_c regalloc tag binding               YES — same emitter path
  spill markers -> spill_pass1/2            YES via the DIT spill
      variants: the generated C contains 33 spill_re references.
  per-position twiddle policy (t1p)         NO — t1-specific; the one
      production lever not wired for hc (candidate, queued)
  log3 / twidsq variants                    N/A (t1-specific math)
  codelet_oop UG/UL machinery               N/A (different emitter/ABI
      family; hc uses emit_c, as do inplace/strided/trig)

CORRECTIONS on the record:
- Section 65 claimed "spill machinery is builder-side, hc has nothing
  to hook": WRONG. Markers come from the builder; the BOUNDING lives
  in emit_c (regalloc + spill_pass1/2 at emit_c.ml:1039-2199) and
  both now reach hc. I was misled by conditional eprintf logs.
- Section 66 claimed "markers do not engage": WRONG, same cause.
- This section's opening premise ("ours = production-equivalent
  spills, per the 151->78 doc figure") also WRONG: that figure is a
  different family. Production r16_t1_dit_fwd = 18 rsp-vmovs.

THE QUANTIFIED VERDICT (bench_t1_cliff.c):
  production t1_16: 3723 ns, 43.5 GB/s   | t1_4: 286 ns, 136.0 GB/s
  -> the radix-16 per-byte cliff is 3.1x IN PRODUCTION, intrinsic to
  the monolithic 16-point butterfly on this uarch. Our hc2hc_16
  (38 GB/s, 3.7x cliff, 79 spills) sits ~15%/byte behind production
  at the same radix. The 79-vs-18 spill delta has a coherent
  mechanism: the sym wrap runs AFTER the spill builder bound its
  pass clusters, and sym1's mirror reversal (im[k] <- im[n-1-k])
  drags values across the pass boundary, re-inflating live ranges.
  Named fix: mirror-paired pass-2 cluster scheduling inside the
  builder (so k and n-1-k co-complete). Ceiling if fully closed:
  ~4-6us per (16,16)-class plan — REAL but second-order behind the
  context wall (section 66), so it queues behind prefetch/scheduling.

## 67. The context-wall attack: planned, executed, falsified, closed

Plan of record: docs/context_wall_plan.md. The mechanism model
(per-row streamer warm-up, ~2 misses x 4r short rows per call) fitted
all three measured regimes and produced two pre-registered levers; a
third ("column scheduling for line reuse") was WITHDRAWN at planning
time — columns touch disjoint rows, there is nothing to reuse.

EXECUTED:
E1 software prefetch (next column's 4r row starts, lines 0+8):
   MEASURED NEGATIVE, 3-5% across plans (A/B: (4,4,16) 109.5 off vs
   113.2 on; (16,16) 84.9 pre vs 90.1 with). On this 1-vCPU KVM the
   prefetches contend for the fill buffers the demand stream needs.
   Default flipped to opt-in (VFFT_RFFT_PREFETCH); note in rfft.h.
E2 Q-fold restored where legal (Kb == K): slot streams lengthen from
   2KB to Q*2KB on Q>1 stages. Worth ~6us at (4,4,16); kept.

LEDGER #25: both pre-registered bands missed; E1+E2 < 10us combined
fired the plan's own falsification clause. The cold-start model is
wrong at the executor-schedule level ON THIS CONTAINER; the residual
warm-vs-pipeline gap is not addressable by instruction scheduling
here. All 14 plan gates PASS throughout.

CLOSING POSITION (container-side native-rfft tuning CLOSED):
  best plans: (8,32) 80.7us / (16,16) 84.9-87.5 across runs
  vs half-complex in-place 76.8 | vs MKL ~56 (same-env, +-25% noisy)
The native path sits at 0.88-0.95x of the wrapper it replaces, on a
VM whose memory subsystem has now eaten three falsified mechanisms
(capacity, aliasing, cold-start). Per the plan's pre-commitment, the
next instrument is REAL HARDWARE (i9-14900KF / EPYC 9575F) with perf
counters — where the prefetch and fold verdicts may invert, and where
the global #1 queue item (stride-wisdom regen, +50%) already lives.

WHAT REMAINS IN P2 proper (functional, not perf): the hc2c
natural-split terminator (design D2) replacing the d=0 packed store —
then P3 three-way benchmarking belongs to real metal alongside the
queued second-order items (mirror-paired spill scheduling ~4-6us,
per-pos hc twiddles, r16 mid GEMV, leaf streams).

## 68. FFTW rdft source comparison (code-level, no benchmarks)

Full writeup: docs/fftw_rdft_comparison.md. Headlines:
- Their khc2hc codelet walks the ENTIRE column range in one call
  (m-loop inside, twiddles as a contiguous linear stream, mirror
  pointer descending) — the call-granularity inverse of our
  per-column ABI, aimed exactly at where our context wall lives.
- Their executor decomposition (cld0 k=0 child / one ranged interior
  call / cldm mid child, twin mirror base pointers) is ISOMORPHIC to
  what sections 60-62 derived independently. The math agrees
  everywhere; only execution strategy differs.
- Both real-path buffered solvers copy column batches to a
  CONTIGUOUS buffer with a deliberately NON-pow2 pitch ("should not
  be 2^k to avoid associativity conflicts") — FFTW hit our wall and
  engineered around it by changing the access pattern; they use no
  prefetch intrinsics anywhere in the rdft path. Consistent with our
  E1/blocking falsifications.
- SIMD axes: their core rdft codelets are SCALAR+ILP; their only
  vectorized real-path family (hc2cfdftv) packs ADJACENT COLUMNS
  into lanes; ours packs the K-batch. For K=256 ours matches the
  problem; their per-transform working set (~4KB) is cache-resident
  by construction — the deep reason the wall bites our schedule.
- dft-r2hc.c is our half-complex wrapper, living inside FFTW as a
  planner alternative. Keep ours; let wisdom choose (T3).
Queue updates: T1 ranged multi-column codelets (generator, ABI v3)
and T2 dobatch copy-to-contiguous (executor) added as the two
precedent-backed levers for the REAL-HARDWARE session; T3 planner
policy noted for the wisdom layer.

## 69. D2 shipped: natural-split terminator — P2 FUNCTIONALLY COMPLETE

Design of record in docs/native_rfft_design.md (D2 final). The piece
of theory that made it one-pass buildable: the CONSTANT-BOUNDARY
LEMMA — the direct/conjugate-mirror regime boundary s* =
floor(r/2 - k/m) is constant over the whole interior column range
(r even: r/2-1; r odd: (r-1)/2), so the FFTW-khc2c-shaped 4-pointer
slot map (Rp/Ip ascending, Rm/Im mirror) bakes into the codelet at
generation time. The existing sym conjugation boundary coincides
with s* for both parities (same "frequency exceeds N/2" line) — no
math-layer changes.

SHIPPED:
- Emitter: hc2c_natural mode (sub-mode of hc_strided; overrides
  signature + output slot map only). gen_main --hc2c-nat; spill
  dispatch + recipe trigger included; naming radix{R}_hc2c_nat.
- Coverage: hc2c_nat for r in {2,3,4,5,7,8,16}, both ISAs. Tree
  825 + 72 = 897 files; 825 byte-identical (leakage gate).
- Codelet formula gate (4-pointer placement + conjugation sign):
  7/7 PASS, zero fix iterations. The gate also pins the natural
  convention: the Im array receives Im(conj G) for uppers — i.e.
  out_im row N-f holds Im X[N-f]. benchmarks/gate_hc2c_nat.c.
- Executor: rfft_execute_fwd_natural(p, x, out_re, out_im) —
  (N/2+1) x K planes. Packed cascade for d >= 1 (full-width fold);
  terminator: k=0 via r2cf into a scratch column + row scatter
  (im[0] = im[N/2] = 0 handled there), interior via hc2c_nat (one
  call covers residues k and m-k), mid via the SHARED s-blocked
  kernel (rfft_mid_column, mode flag packed/natural — the packed
  executor and both natural sites now use one implementation; my
  first cut had scalar natural mids, +27us at (16,16), caught by
  same-run bench and fixed by the refactor).
- End-to-end gate vs unpacked packed reference: 12/12 cells PASS at
  machine epsilon (several bit-identical). Packed 14-cell suite
  re-PASSED after every header change. benchmarks/gate_rfft_natural.c.

NUMBERS FOR THE RECORD (same-run, this container, K=256 N=256):
  (8,32) : packed 81.1us | NATURAL 72.5us
  (16,16): packed 98.2   | natural 94.0
  (4,4,16): packed 114.9 | natural 117.2
The natural path beats packed where it matters: the terminator
writes (N/2+1) rows, not N — half the d=0 store traffic, a
STRUCTURAL win. Against the half-complex wrapper (76.8us historical
best): the native natural path at 72.5 is parity-to-ahead even on
the container that ate every scheduling lever. P3's three-way
(native vs wrapper vs MKL) belongs to real metal as planned, but the
functional story is closed: leaf -> combine -> natural terminator,
end to end, gated.

## 70. P3 chapter opened: T1 shipped, T2 designed, metal kit ready —
## and the layout datum

### T1 (ranged multi-column codelets) SHIPPED AND GATED
Emitter hc_ranged mode: the column loop lives inside the codelet
(FFTW khc2hc structure) — outer kc-loop wraps the v-loop, parameter
pointers bump per column (in ascending / im descending / out pair /
tw += r), render layer untouched. New families radix{R}_hc2hc_dit_rng
and radix{R}_hc2c_nat_rng, 7 radices x 2 ISAs (tree 825+100=925; 825
byte-identical). Codelet gates: 14/14 PASS, zero codelet fixes — the
one failure was MY gate's re-derived sym2 boundary (2i < n is the
source-of-truth condition; even radices exposed the off-by-one at
slot n/2; lesson re-learned: copy proven reference math, never
re-derive it in a gate). Executors: VFFT_RFFT_RANGED compile switch
in packed interior, natural middle stages, and natural terminator;
all four gate configurations (packed/natural x off/on) ALL PASS.
Same-run container record (non-binding): (8,32) natural 71.1us
ranged vs 72.5 per-column.

### T2 (dobatch copy-to-contiguous): DESIGNED, NOT BUILT
docs/t2_dobatch_design_note.md — the FFTW trick is NOT a
transliteration (their non-pow2 pitch lives on the column axis their
codelet iterates; ours iterates lanes). Resolution: pad the SLOT
axis (vl + pad, non-pow2) first, optionally column batching on top;
break-even model vs the measured context penalty included. Metal
decides; correctness gating trivial once built.

### P3 metal kit: benchmarks/p3_metal/
bench_p3.c (factorization sweep, natural + packed lanes, FFTW
plan_many lane, MKL real-CCE lane, every vfft lane self-validated
against FFTW before timing, CSV output), build.sh (four variants:
base/ranged/prefetch/both), run.sh (taskset pinning, optional perf
counters), README (methodology + the container verdicts metal must
re-adjudicate: ranged, prefetch, Kb, factorization, the MKL gap).
Wrapper lane left as a documented TODO (needs main-library objects).

### THE LAYOUT DATUM (container smoke, same-run, lane-major
### N=256 K=256, stride=K dist=1 — the product layout)
  vfft natural (8,32):  81.9us
  MKL real CCE       : 105.9us   -> vfft leads 1.29x SAME-RUN
  FFTW plan_many     : 324.1us   -> 4.0x
Caveats, in full: container MKL noise is the known +-25%, but the
1.29x is a same-run ratio (the trustworthy kind); the historical
"MKL ~56us" was MKL's PREFERRED layout (contiguous per transform),
not this one. The honest statement: in the lane-major batched layout
— the shape the K-parallel-series consumer actually wants — our
batch-lane SIMD axis is native and MKL's within-transform axis pays
the strided access; section 68's SIMD-axes taxonomy, measured. FFTW's
4x deficit confirms the same analysis (vrank one-at-a-time, scalar
rdft codelets). Metal will issue the binding numbers, including
MKL-preferred-layout lanes for the full apples/oranges matrix.

## 71. Un-cliffing big radices: the measured ladder (user question)

The r16 metadata named the suspect (memory_floor 158 vs essential 94,
+68% port traffic from the cross-pass cut) — and the experiments
DEMOTED it. Hot r16_t1 ladder, same harness, bit-correct outputs:

  production (recipe cut)   43.5 GB/s   18 spills
  E1 no-cut (--no-recipe)   45.6        19 spills   (+5%)
  E2 split-radix            45.7        13 spills, 230 flops (+5%)

Both bands grazed at the low end: the plateau survives schedule AND
topology changes, so neither the cut nor op count dominates. The
counters say what does: ~0.63 FP IPC and 0.27 mem-ops/cyc against
~3/cyc capability — the codelet is LATENCY-BOUND on the DFT-16
network depth x FMA latency, with nothing independent in flight.

THE lever with real headroom (named, unbuilt here): COLUMN
INTERLEAVING — schedule two independent columns' butterflies
interleaved per iteration to fill the latency slots. This is
Tugbars's own R=20 IL technique (2 sub-FFTs, 24 ZMM) from the main
repo; this generator has no IL mode yet. Register math says it fits:
2 x peak_live 12 = 24 <= 28 budget at r16. It composes naturally
with the T1 ranged codelets (unroll the kc loop by 2, interleave
bodies). Generator build item, queued at the top of the codegen
ladder.

E3 plan space (1024, same harness, benchmarks/bench_1024_planspace.c):
  (4,16,16) baseline  715.4 us
  (32,32)             602.9 us   <- -16%, band [550,850] HIT
  (8,8,16)            705.0      (16,64) 735.9  (64,16) 760.0
  (4,4,64)            812.6
(32,32)'s 2-stage traffic (16MB vs 24MB) beats three-stage despite
r32 codelet quality — moving c2c1024 from 0.78x to ~0.92x of MKL
with ZERO new code. Action: wisdom-entry change (1024@K256 ->
(32,32)); strengthens the i9 wisdom-regen item — the in-container
exhaustive now disagrees with the wisdom file at BOTH 1024 rows.

Split-radix note: best codelet variant found (fewest spills, fewest
ops, +5%), adoptable suite-wide via the existing env gate after a
proper gating pass; queued behind IL.

## 72. IL probes: two falsifications that converge on the R=20 design

User context integrated first: the SR construction here is a SKETCH
(his words) next to a tuned CT — which makes section 71's result
(sketch SR ties tuned CT single-stream: 45.7 vs 43.5, 13 vs 18
spills, 230 vs 244 flops) a strong argument FOR SR maturation, not a
+5% shrug.

PROBE 1 (compiler-level IL2, two inlined copies): 44.0 GB/s — MISSED
[50,85]. INVALID as a test: every emitted temp is pinned with asm
volatile, which gcc cannot reorder across. Our own anti-scheduling
device blocks compiler interleaving by construction.

PROBE 2 (generator-level IL2, dft_expand_twiddled_il2: two DAG
instances concatenated, SU braids by readiness; --il2 flag; both
topologies; outputs bit-match 2x production):
  2x production sequential   43.2 GB/s
  CT-IL2 braided             31.2        (106 spills, peak_live 130)
  SR-IL2 braided             39.7        ( 98 spills, peak_live 128)
MISSED [55,90], NEGATIVE. Mechanism: naive whole-instance braiding
doubles the monolithic live set; the 28-register budget converts the
intended latency overlap into spill traffic. SR loses LESS (narrower,
as predicted) but both lose.

WHAT THE TWO MISSES CONVERGE ON: register-disciplined, PASS-PAIRED
interleaving — exactly the user's R=20 IL design (per-pass peak 12,
two instances paired at 24 ZMM, 8 free). The cross-pass cut that
section 71 demoted (+5% when removed, single-stream) is REHABILITATED
as the enabler: the cut bounds per-pass live to ~12 so two instances
fit, and the cut's memory traffic overlaps the partner instance's
compute. Build item, sharpened: braid dft_expand_twiddled_spill
instances PASS-WISE (pass1A+pass1B, then pass2A+pass2B), not
whole-DAG. For SR: the maturation target is pass-STRUCTURED SR
(E8/O1/O3 blocks scheduled with per-block live <= ~12-14) so SR
hosts pairing with less cut traffic than CT's full 16-slot cut —
the sketch's narrowness (13 spills, fewest ops) is the evidence the
shape supports it.

Standing answer to "IL vs split": neither alone. Pass-paired IL is
the multiplier; SR is the better pass structure to pair. Sequencing:
(1) pass-paired IL over the EXISTING CT recipe (machinery present,
markers + ct already computed); (2) SR recipe topology (the dft.ml
follow-up PR) bringing the sketch to parity; (3) pair over SR.
All three adjudicate finally on metal.

## 73. The scheduler is NOT the bottleneck: codelets are THROUGHPUT-bound (measured)

User question: near-min flops, low spills, low max-live, no zmm spill-stores -> how can we
lose to MKL? Better ILP? schedule.ml SU-for-DAGs weakness?

MEASURED (critical-path analysis of generated codelets vs throughput floor vs runtime):
| codelet   | crit-path(cyc) | arith | floor=arith/2 | measured(K8,/call) | bound       |
| CT-16 n1  | 31             | 144   | 72            | ~71                | THROUGHPUT  |
| CT-32 n1  | 27             | 386   | 193           | ~192               | THROUGHPUT  |
| CT-64 n1  | 35             | 978   | ~489          | ~391 (gcc fuses)   | THROUGHPUT  |
| SR-16     | 35             | 167   | 84            | ~73                | THROUGHPUT  |
| SR-16 fma | 35             | 152   | 76            | ~69                | THROUGHPUT  |

VERDICT: critical path is ~30 cyc and FLAT in N (dep depth ~log N: 4->6 stages); work grows
as N log N so the throughput floor climbs and measured tracks it. CT-64: 35-cyc crit-path vs
~400-cyc runtime = >10x ILP slack, FULLY exploited (both FP ports saturated, ~2 FP ops/cyc).

=> NOT latency/ILP/schedule-bound at small K. The cp-list scheduler (cp_dist primary, su
   tiebreak) already saturates the 2 FP ports; SU-for-DAGs is moot because the schedule is
   not the binding constraint. A scheduler rewrite recovers ~nothing here.

THE TWO REAL BOTTLENECKS (neither is the scheduler):
  - small K / L1-hot: 2-FP-PORT INSTRUCTION THROUGHPUT. Only lever = fewer instr/flop = more
    FMA fusion (drops the floor). Bounded: flops near-min + gcc -ffp-contract=fast already
    fuses most. (forced-lift's -4.5% is this lever, and its ceiling.)
  - large N / cache-resident: MEMORY BANDWIDTH (the 43.5 GB/s radix-16 cliff, sec 71). This
    is the "0.63 FP IPC" regime - ports idle waiting on data, NOT compute. Lever = fewer
    passes / less traffic (large radix, the (32,32)-for-1024 win; prefetch).

MKL TIE-IN: in MKL-preferred layout MKL's edge is data MOVEMENT not compute (our compute is
port-bound and tied); in batched layout we win 1.29x by skipping MKL's 167 permutes (sec in
mkl_internals_findings.md). The SR work and any scheduler work are orthogonal to the MKL gap.
