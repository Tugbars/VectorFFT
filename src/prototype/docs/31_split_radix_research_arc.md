# 31. Split-Radix Research Arc: Algsimp Is the Production Path

## Context

VTune analysis of N=131072 K=4 (`docs/dev/vfft_n131072_vtune.md`)
showed VectorFFT executes 41.24B instructions vs MKL's 16.95B for the
same FFT — a **2.43× instruction-count gap** that nearly cancels with
our 2.68× higher IPC, yielding a 1.17× wall-time margin. The leading
hypotheses for the instruction gap were:

1. **Split-radix algorithm.** Classical result (Yavne 1968, Sorensen
   et al. 1986): SR reduces multiplication count by ~33% vs vanilla
   Cooley-Tukey for power-of-two N. genfft's `newsplit` pass is the
   reference implementation.
2. **Larger codelets** (e.g. R=128 monolithic) reducing per-stage
   overhead.
3. **Hand-tuned assembly** giving denser scheduling than compiler
   output.

This doc records the experimental arc that investigated SR as a
v2.0 lever. The headline finding is negative for the original
hypothesis but positive for what it tells us about our pipeline:
**in CSE-heavy codelet generation, algorithmic substitution does
not translate to runtime wins because the simplifier converges
different starting points to equivalent local optima.**

## What got built

`lib/split_radix.ml` (~200 lines) implements the standard SR
decomposition:

```
For N = 4M, k in [0, M):
  T1[k]      = W_N^k    · O1[k]                           (twiddle)
  T3[k]      = W_N^{3k} · O3[k]                           (twiddle)
  U[k]       = T1[k] + T3[k]                              (shared sum)
  V[k]       = T1[k] - T3[k]                              (shared diff)
  X[k]       = E[k]    + U[k]
  X[k+M]     = E[k+M]  + (-j) · V[k]                      (forward)
  X[k+2M]    = E[k]    - U[k]
  X[k+3M]    = E[k+M]  + (+j) · V[k]
where:
  E   = DFT_{N/2} over even-indexed inputs x[2n]
  O1  = DFT_{N/4} over stride-4 inputs x[4n+1]
  O3  = DFT_{N/4} over stride-4 inputs x[4n+3]
```

A `Split_radix` variant was added to the `algorithm` type in
`lib/dft.ml`. Routing is opt-in via `VFFT_SPLIT_RADIX=1`: when set,
pow2 N ≥ 8 routes to SR. The recursion bottoms cleanly at N=4
(CT(2,2)) and N=2 (Direct).

Cross-module mutual recursion uses a callback parameter:
`dft_split_radix` takes `~dft_rec:(int → ...)` so SR can recurse
on its sub-DFT inputs through the picker, dispatching to SR/CT/Direct
based on each sub-size.

The recipe machinery (cluster-sequential spill at the PASS 1 / PASS 2
boundary) does not generalize to SR's three-sub-DFT topology, so SR
falls back to the plain non-recipe path. A follow-up doc would design
SR-aware cluster boundaries; we declined to do that work given the
findings below.

## Subsidiary fix during build-out

Wiring SR exposed an existing latent bug. The `lift_sub_neg_mul` pass
(see [30_sub_neg_mul_fnmsub.md](30_sub_neg_mul_fnmsub.md)) had been
running as a standalone post-pass, rebuilding the algsimp DAG to
rewrite `Sub(Neg(Mul(a,b)), c)` → `NK_Fma(a, b, c, true, true)`. This
worked for R=25 but silently broke R=32 and R=64 CT codelets, which
emitted asm referencing undefined variables (`t368`, `t908`, …).

Root cause: spill markers reference algsimp tags via hashcons. When
`lift_sub_neg_mul` ran as a post-pass, it created new Fma nodes with
new tags. The old tags remained in the hashcons table but were no
longer reachable from any output assignment. Spill markers captured
*before* the rewrite still pointed at the old tags. emit_c then saw
markers referencing nodes that no consumer drove → spill_load with no
matching spill_store.

Fix: move the rewrite into `mk_sub_binary` as a smart-constructor
peephole. The rewrite now fires at construction time (during
dedup_sub_pairs' rebuild) so the resulting DAG has consistent tags
throughout. Spill markers, captured later via `lift_spill_markers`'s
`of_expr` + hashcons, naturally find the post-peephole nodes.

This is a general lesson worth repeating: **IR transformations that
interact with downstream tag references should be peepholes during
hashcons construction, not post-passes that orphan old tags.**

Verification post-fix:
- R=32, R=64 CT codelets compile clean
- R=25 vxorpd count remains 0 (doc 30 win preserved)
- R=25 IR stats unchanged: 6 Fmas, 666 vector instructions
- Prime correctness: 56/56 PASS

## The four-layer diagnostic

The investigation drilled down through four layers, each answering
a different question.

### Layer 1: Pre-algsimp IR (raw + post-hashcons unique nodes)

Walks the raw `Expr.assignment list` from `dft_expand_twiddled` and
counts (a) textual nodes (over-counts shared subtrees) and (b)
unique-after-hashcons nodes. The "unique" count is what algsimp sees
as input.

```
Pre-algsimp unique nodes (post-hashcons, pre-simplification):
R    CT muls   SR muls   SR savings
8       236       132        -44%
16      636       324        -49%
32     1596       660        -59%
64     3836      1444        -62%
```

**SR is producing dramatically less raw multiplication than CT.**
In fact the savings exceed the textbook 33% figure — the SR
decomposition exposes additional twiddle-folding opportunities that
the CT structure doesn't. This confirms the SR construction is
correct and structurally distinct from CT.

### Layer 2: Post-algsimp op count

The same DAGs after the full simplification pipeline
(`factor_common_muls`, `factor_by_atom`, `dedup_sub_pairs`,
`share_subsums`, FP transpose loop):

```
Post-algsimp mul-class instructions:
R    CT     SR     SR/CT     Δ
8    33     33     1.000     0
16   86     86     1.000     0
32   218    213    0.977    -5
64   518    510    0.985    -8
```

**Op counts converge.** The 44-62% raw mul savings collapse to
0-1.5% post-algsimp differences. CT's denser raw IR has more
redundancy for our passes to extract; SR's sparser raw IR has the
sharing structurally encoded. Both starting points reach approximately
the same canonical post-algsimp form.

This is the central finding: **our algebraic simplifier is a strong
normalizer.** `factor_common_muls`, `share_subsums`, and the FP
transpose loop together find essentially all the algorithm-level
sharing that distinguishes SR from CT, regardless of which
construction provides the input.

The reduction ratios are illuminating:
- R=64 CT: 3836 raw muls → 510 mul-class (7.5× compression)
- R=64 SR: 1444 raw muls → 510 mul-class (2.8× compression)

Both converge to the same destination by different routes.

### Layer 3: Post-algsimp structural diff

Op-count equality is not structural identity. A canonical
S-expression fingerprint for each output assignment, compared per
output bin:

```
R    Outputs   Identical  Differ   % differ
8       16        8         8       50%
16      32       16        16       50%
32      64       16        48       75%
64     128        8        120     94%
```

**At R=64, 94% of post-algsimp output DAGs are structurally distinct
between CT and SR.** Despite equal op counts, the trees that produce
each X[k] are different.

The bins that match are exactly the algebraically simple ones — those
whose twiddle vectors collapse to roots of small-N unity:
- R=8 identical: bins 0, 2, 4, 6 (W_8 powers reducing to W_4)
- R=32 identical: bins 0, 4, 8, ..., 28 (W_32 powers reducing to W_8)
- R=64 identical: bins 0, 16, 32, 48 (W_64 powers reducing to W_4)

For these "smooth" bins, both algorithms produce the same canonical
simplified form because there's not much to differ over. The bulk
of the actual computation (94% at R=64) produces structurally distinct
DAGs whose S-expression serializations are also notably **shorter**
for SR (-14% to -38% at R=64).

### Layer 4: Asm metrics and runtime

Asm-level static metrics from gcc-13 -O3 -mavx512f -march=skylake-avx512:

```
R    Variant  FP-instr  vmovapd  stack-mem  recipe?
8    CT       77        3        0          yes
8    SR       76 (-1)   2 (-1)   0          NO
16   CT       211       16       12         yes
16   SR       207 (-4)  12 (-4)  6 (-50%)   NO
32   CT       591       106      86         yes
32   SR       567(-24)  88 (-18) 74 (-14)   NO
64   CT       1485      318      263        yes
64   SR       1608(+8%) 441(+39%) 389(+48%) NO
```

CT codelets use the spill recipe (the cluster-sequential PASS 1 /
PASS 2 boundary); SR codelets fall back to the plain expansion. So
at R≤32, SR shows fewer instructions and less stack memory traffic
*despite not having a recipe*. At R=64, the recipe is essential and
SR's lack of one shows up as +39% vmovapd and +48% stack ops.

Runtime, best-of-7-trials, virtualized SPR class CPU at 2.1 GHz fixed,
wall-clock ns per call (compute-dominated regime, K ≤ 64):

```
R     SR/CT median K=8-64
8     1.00-1.06          parity to slight SR loss
16    0.94-1.07          mixed
32    1.10-1.26          SR consistently 10-26% slower
64    1.13-1.52          SR consistently 13-52% slower
```

**The asm-level metrics did not predict runtime.** The R=32 case is
the cleanest disconnect: SR has 24 fewer FP instructions, 18 fewer
vmovapd, and 12 fewer stack ops, yet runs 10-26% *slower* in the
compute-dominated regime.

Plausible explanations for the disconnect, none confirmed:
- Different critical path length (longer dependency chains in SR
  even with fewer total ops)
- Different register allocation outcomes; the saved vmovapd may be
  replaced by costlier stalls
- Different µop fusion / port pressure; same FP instruction count
  can hit different execution port distributions

In the memory-dominated regime (K ≥ 256), SR occasionally wins —
R=16 K=2048 shows 0.80-0.94 SR/CT. But this regime is where the
codelet is rarely the bottleneck anyway.

## The cooperation question

Given that CT and SR have structurally distinct post-algsimp DAGs,
we tested whether they could "cooperate" — would running our pipeline
on the *union* of CT and SR constructions yield meaningful additional
sharing?

`bin/sr_union_probe.ml` emits both X_CT[k] and X_SR[k] as separate
output assignments and runs the full pipeline:

```
R    CT alone  SR alone  Union   Sum (CT+SR)  Union/Sum  Union/smaller-alone
8       84        84      104       168         0.62       1.24
16     227       227      263       454         0.58       1.16
32     582       575      818      1157         0.71       1.42
64    1412      1396     2177      2808         0.78       1.56
```

The union is 22-42% smaller than the naïve sum, but **always larger
than either alone** (16-56% larger). The sharing is concentrated at
the bottom of the DAG (input cmuls — `cmuls=14` in R=8 CT alone, SR
alone, AND union). High-level computation (Muls, Fmas, output
butterflies) stays disjoint between constructions.

Implication: hashcons already provides the cross-construction sharing
that's free. To exploit deeper cooperation would require recognizing
that X[k]_CT and X[k]_SR are mathematically equivalent (not just
structurally identical) — which is e-graph saturation territory
(`egg` in Rust). Implementing e-graph rewriting in OCaml is a major
project with uncertain gain ceiling. **Not pursued.**

## Conclusions

**Algsimp is the production path.** Algorithm choice (CT vs SR vs
CPSR vs Bruun) is a *starting condition* the pipeline normalizes.
The post-algsimp form is determined by the simplifier's pass set,
not by which decomposition fed it. This explains why MKL's 2.43×
instruction-count gap is unlikely to be primarily algorithmic: even
if MKL uses SR, our pipeline would converge from CT to a similar
density anyway. The remaining hypotheses for MKL's gap — denser
codelet structure, hand-tuned assembly, codelet fusion across stages
— are more credible.

**Asm metrics are a misleading runtime predictor.** Static FP
instruction count, vmovapd count, and stack-memory operation count
each correlated negatively or weakly with measured runtime in this
investigation. For codelet generation work, runtime measurement is
the ground truth; asm metrics are informative for understanding
*why* something is fast but not for predicting whether it will be.

**The peephole-vs-post-pass distinction matters.** IR rewrites that
must remain consistent with downstream tag references (spill markers,
schedule annotations) belong in smart-constructor peepholes during
hashcons construction. Standalone post-passes that walk and rebuild
the DAG silently orphan tag references whose holders haven't been
notified.

**SR construction stays in the codebase.** It's correct, well-tested,
opt-in, and zero-cost when not enabled. It serves as:
- A research touchpoint for future work (CPSR, e-graph experiments)
- A cross-check for algsimp regressions (if SR and CT diverge in op
  count, something changed in the pipeline)
- An alternative IR shape for testing recipe machinery on different
  topologies

The diagnostic tools (`bin/sr_diag.ml`, `bin/sr_structural_diff.ml`,
`bin/sr_union_probe.ml`) are kept in the bin directory — the
methodology is reproducible if the same questions resurface.

## What this redirects to

If SR is not the v2.0 lever, the remaining attackable items in
priority order are:

1. **Codelet fusion across stages.** From the VTune doc: 21% of CPU
   time at N=131072 K=4 is `_stride_execute_fwd_slice_from` —
   per-stage executor overhead paid 8 times across the planned
   factorization. Fusing the 8 stages into one monolithic codelet
   eliminates the overhead. This is *concrete* slice of waste,
   independent of algorithm choice, and the largest unexploited win
   we know exists.

2. **Algsimp pass profiling for R=128 scaling.** The IR pipeline
   currently doesn't terminate in reasonable time past R=64
   (approximately O(N⁴) scaling somewhere — likely
   `share_subsums` or the FP transpose loop). Fixing this unlocks
   larger codelets and the experiments they'd enable.

3. **Recipe machinery generalization.** The current
   PASS 1 / PASS 2 boundary assumes CT structure. Even if SR isn't
   pursued, generalizing the cluster boundary lets us experiment with
   other decomposition shapes (mixed-radix at large N, fused
   monolithic codelets).

## Reproducing

```
# IR raw counts (hypothesis 1: SR construction works)
./_build/default/bin/sr_diag.exe 64
VFFT_SPLIT_RADIX=1 ./_build/default/bin/sr_diag.exe 64

# Structural fingerprint diff (hypothesis 2: post-algsimp diverges
# despite equal op count)
./_build/default/bin/sr_structural_diff.exe 64

# Union probe (hypothesis 3: cooperation potential)
./_build/default/bin/sr_union_probe.exe 64

# Generated codelets with each variant
./_build/default/bin/gen_radix.exe 32 --twiddled --in-place --emit-c
VFFT_SPLIT_RADIX=1 ./_build/default/bin/gen_radix.exe 32 --twiddled --in-place --emit-c
```

## See also

- [30_sub_neg_mul_fnmsub.md](30_sub_neg_mul_fnmsub.md) — the doc-30
  fix this work generalized into a peephole
- `docs/dev/vfft_n131072_vtune.md` — the VTune deep-dive that
  motivated this investigation
- `lib/split_radix.ml` — the construction
- `bin/sr_diag.ml`, `bin/sr_structural_diff.ml`,
  `bin/sr_union_probe.ml` — diagnostic tools
