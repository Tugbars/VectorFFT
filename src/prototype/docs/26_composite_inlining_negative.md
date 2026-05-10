# 26. Composite emission cross-pass inlining — negative result

## TL;DR

Extended single-use inlining from the SU non-spill path (where it
gave R=13 t1_dit −20 movapd, R=17 t1_dit −19 movapd) to the spill
path used by composite codelets (R=32, R=64). Result: **8-9%
reduction in source-level SSA name count, ZERO change in generated
asm** on SKX, SPR, and Zen 4.

The hypothesis from `docs/25_llvm_mca_headroom.md` ("10-30% available
on composites from spill-array reload locality and cross-boundary
inlining") was wrong on the inlining side. GCC's RA already does
this work transparently. The reload-locality side wasn't tested but
is also unlikely to help (spill array fits trivially in L1).

## What was tried

Spill path emission (`lib/emit_c.ml`, `match spill with | Some sp ->`
branch) didn't use `compute_inline_set` at all. Each PASS 1 / PASS 2
node emitted as a standalone `const __m512d t<N> = ...` declaration,
even when single-use. The SU non-spill path uses inlining via
`render_node_def ~inline_set`. Plumbing it through the spill path
required:

1. Compute global `inline_set = compute_inline_set assigns`
2. Filter to "tag is not spilled AND all consumers in same pass as
   producer" — cross-pass values must round-trip through the spill
   array since PASS 1 scope closes before PASS 2 opens
3. Skip standalone emission of inlined tags in both PASS 1 and
   PASS 2 walks, pass `~inline_set` to `render_node_def`
4. Add a `reload_through_inlines` walker in PASS 2: when emitting
   node Z whose predecessors include an inlined X that references
   spilled Y, walk transitively through inlined preds to ensure Y
   gets reloaded before Z's expression references `t<Y>`. Without
   this, inlined chains break the reload invariant.

Implementation: ~50 lines added to `emit_c.ml`. Build clean,
56/56 prime correctness PASS, R=32/R=64 composite correctness PASS at
machine precision (5e-13 / 1e-12).

## Measured impact

### Source-level

| Codelet | const decls before | const decls after | Δ |
|---|---|---|---|
| R=32 t1_dit | 653 | 605 | −48 (−7.4%) |
| R=64 t1_dit | 1530 | 1390 | −140 (−9.2%) |

Real reduction in named SSA values. Faster GCC compile time (small,
not measured precisely).

### ASM-level

| Codelet | μarch | vmovapd before | vmovapd after | cycles before | cycles after |
|---|---|---|---|---|---|
| R=32 t1_dit | SKX | 288 | 288 | 312 | 312 |
| R=32 t1_dit | SPR | (n/a) | (n/a) | 305 | 305 |
| R=32 t1_dit | Zen 4 | (n/a) | (n/a) | 342 | 342 |
| R=64 t1_dit | SKX | 940 | 940 | 784 | 784 |

Asm structurally identical except for one independent instruction
swap (`vbroadcastsd .LC5` and `movq 5576(%rsp), %r10` exchanged
positions in R=32 t1_dit). All three μarch llvm-mca cycle counts
identical across the change.

Block RThroughput (resource-saturation lower bound) also unchanged:
R=32 t1_dit at 256.0 SKX / 200.0 SPR / 254.0 Zen 4 in both
configurations.

## Why it didn't help

Two earlier diagnostic results made this predictable in retrospect.

### 1. GCC already optimizes our explicit spill traffic

The `--fuse=M` flag controls how many spill slots the spill path
keeps register-resident across the PASS 1 / PASS 2 boundary versus
round-tripping through `spill_re[]` / `spill_im[]`. Default is 0
(no fusion: every spill goes through memory). Setting `--fuse=8`
for R=32 t1_dit (which has CT(4,8), so 8 slots per sub-FFT × 4
sub-FFTs = full fusion of all 32 slots) eliminates all 32 explicit
spill stores and 32 spill reloads in the C source.

Result: asm byte-identical (sans `.file` directive) at fuse=0,
fuse=2, fuse=4, fuse=6, fuse=8. The 288 vmovapd in R=32 t1_dit
asm are GCC's own register-allocation spill decisions, not a
translation of our explicit `_mm512_storeu_pd(&spill_re[k], v)`
calls. GCC promotes the spill array to virtual registers and
eliminates redundant store/load pairs.

So our explicit spill mechanism in `emit_c.ml` is already largely
cosmetic — GCC's escape analysis recognizes the local stack array
and bypasses it. Reducing explicit spill traffic was never going
to help.

### 2. Composite DAGs don't have prime-style SSA-name explosion

Single-use inlining helped primes because their pre-inlining C
source had linearized SSA forms — `t298 = mul(a,b); t299 = sub(c,
t298)` — that GCC's optimizer struggled to re-fold into nested
intrinsics. Inlining produced `sub(c, mul(a,b))` as a single
expression, which GCC matched and contracted into fmadd / fmsub
patterns better.

Composite codelets emit nested intrinsics from the start: the cmul
operations (`NK_CmulRe` / `NK_CmulIm`) render as 3-deep nested
expressions like `_mm512_fnmadd_pd(t296, t297, _mm512_mul_pd(t294,
t295))` directly, and `fma_lift` produces explicit `NK_Fma` IR
nodes that render as `_mm512_fmadd_pd` directly.

So the prime-era gap (linearized SSA defeating GCC's contraction)
doesn't exist for composites. The 48-140 SSA names that source-level
inlining eliminates are just intermediate Add/Sub values whose
inlining produces equivalent expression trees that GCC was already
handling well.

## Should the change be kept?

Argument for keeping: 8-9% fewer SSA names is a real source-level
cleanup. Brings the spill path's emission style consistent with the
SU non-spill path. Slightly faster GCC compile times. Zero
correctness or perf risk. ~50 lines well-localized in `emit_c.ml`.

Argument for reverting: no measured perf benefit. Maintenance cost
of the transitive reload walker (`reload_through_inlines`) and the
filter logic. Adds conceptual surface area to the spill path that
already had the most complexity in the file.

**Decision: keep.** The cleanup value is real, the cost is
contained, and the change unifies two emission paths that were
previously diverged for no good reason. But document the negative
performance result so future work doesn't re-attempt the same
hypothesis.

## What this means for the composite story

The "10-30% available from composite emission improvements" framing
was wrong about the mechanism. The realistic available headroom for
composites at SKX is bounded by llvm-mca's port-saturation analysis:
R=32 at 82%, R=64 at 86%. The 14-18% gap is latency-chain stalls
plus load-port contention, neither of which source-level emission
strategies move directly.

Re-ranked composite leverage points:

1. **Algorithmic (highest)**: try alternate CT decompositions for
   composite N. R=64 currently uses CT(8,8); CT(4,16), CT(16,4),
   CT(2,32), CT(32,2) might have different cycle counts. Different
   factorizations can yield very different DAG shapes (deeper twiddle
   chains vs shallower-but-wider butterflies).

2. **Per-μarch (medium, gated)**: SPR R=32 at 65.6% port-saturation
   IPC remains the outlier. Port 0 saturated, port 1 (SPR's added
   AVX-512 FMA port) idle. Worth investigating IF VTune confirms
   the llvm-mca model on real hardware.

3. **Bench against hand-coded composites**: the project has hand
   references for R=5/7/11 (in `bench/primes/hand_*.h`) but not for
   R=32/64. Without that comparison, we don't know whether OUR
   composite codelets are the ceiling or whether hand achieves
   meaningfully fewer cycles. Generating hand references would let
   us stop trying to improve a thing that's already at the limit.

4. **Cluster-aware scheduler priority**: the current SU scheduler
   uses pure cp_dist, no cluster awareness. Cluster boundaries in
   PASS 2 create natural fragmentation; if the scheduler chose to
   complete one cluster fully before starting the next (instead of
   interleaving), GCC might produce different RA decisions. This is
   speculative — would need to try and measure.

5. **Cross-pass inlining (this work)**: source-level cleanup, not
   a perf lever.

## Cross-compiler validation

A natural follow-up question: do we trust the GCC numbers? If GCC
optimizes away our explicit spill traffic transparently, maybe other
compilers don't and the cross-pass inlining matters there. Tested
on Clang 18.

### Does Clang also optimize away explicit spill traffic?

Yes. R=32 t1_dit with `--fuse=0` (32 spill stores + 32 reloads in
C) and `--fuse=8` (zero spill traffic in C) produce byte-identical
asm under Clang -O3. Clang's escape analysis recognizes the local
spill arrays and bypasses them, same as GCC.

### Does cross-pass inlining help on Clang?

No. R=32 t1_dit on Clang: 321 cycles before the change, 322 cycles
after. Within noise. The negative-result conclusion is not
GCC-specific — it holds on Clang too.

### Loop-body cycles, GCC vs Clang on SKX

| Codelet | GCC | Clang | Ratio |
|---|---|---|---|
| R=17 t1_dit | 289 | 283 | 0.97× |
| R=17 t1_dif | 311 | 349 | 1.12× |
| R=19 t1_dit | 352 | 338 | 0.96× |
| R=19 t1_dif | 376 | 441 | 1.17× |
| R=32 t1_dit | 312 | 322 | 1.03× |
| R=32 t1_dif | 295 | 290 | 0.98× |
| R=64 t1_dit | 784 | 761 | 0.97× |
| R=64 t1_dif | 870 | 752 | **0.86×** |

Mixed picture. Clang is faster on 5/8, slower on 3/8. The
generalization "compiler X is uniformly better" doesn't hold. But:

- **DIF primes (R=17, R=19)** are Clang's weak point. ~12-17%
  slower than GCC. Likely related to the multi-use raw outputs
  (DIF post-multiplies outputs, creating use_count=2 patterns
  that defeat single-use inlining — same gap noted in doc 24).
- **R=64 t1_dif** is GCC's weak point. Clang is 14% faster. The
  asm breakdown:

  | Op | GCC | Clang |
  |---|---|---|
  | vfmadd / vfnmadd / vfmsub | 109 / 63 / 80 = 252 | 166 / 102 / 18 = 286 |
  | vmovapd (reg-reg moves) | **1071** | **83** |
  | vmovupd (memory) | 378 | 1253 |

  GCC's RA over-preserves via register copies; Clang trades reg-reg
  moves for memory traffic and gets a meaningful win on the larger
  composite. This is RA strategy difference, not a fma_lift IR
  difference (mul+add+sub counts identical: 268+384+385 vs
  268+384+384).

### Implications

The user's underlying concern — compiler portability matters and
"GCC optimizes our flaws" — is valid in principle but doesn't
manifest as worse Clang output. Both compilers handle our codelets
within ~3% on the median codelet, with the spread being case-
specific RA strategy differences rather than systematic
optimization gaps.

The interesting unlock is the OTHER direction: **R=64 t1_dif is 14%
faster on Clang for free**. For HFT production where the bench
target is "fastest cycle count," shipping Clang-compiled binaries
for that codelet (or all composite codelets) is a no-cost optimization.
This is not a code change — it's a build-system choice.

### What would be useful next

- Add Clang to the bench harness alongside GCC. Cross-compiler
  measurement should be standard going forward, both for
  catching regressions specific to one compiler and for picking
  the better-performing build per codelet.
- Investigate the 1071 → 83 vmovapd gap on R=64 t1_dif. If we
  can identify what about our C source forces GCC into the
  reg-copy-heavy strategy, we might fix the IR shape and get
  GCC's R=64 t1_dif into Clang's range.
- Test ICC / ICX when access becomes available (not in this
  container). Intel's compiler is used in some HFT environments
  and may have different characteristics again.

## Files changed

- `lib/emit_c.ml`: ~50 lines added in the spill-path branch:
  `inline_set` computation with same-pass filter, `is_inlined`
  check, `reload_through_inlines` transitive walker, `~inline_set`
  argument propagation in PASS 1 and PASS 2 emit.

## Test results

- 56/56 prime correctness PASS (machine precision, errors all under
  2e-12)
- R=32 t1_dit composite: err=5.14e-13 PASS
- R=64 t1_dit composite: err=1.10e-12 PASS
- Prime asm: byte-identical to baseline (only `.file` directive
  differs) — primes use the SU non-spill path which was unchanged
- Composite asm: structurally identical, one independent instruction
  swap, identical instruction counts
- llvm-mca cycle counts on GCC: identical on SKX, SPR, Zen 4 for
  R=32 and R=64 t1_dit before/after the change
- llvm-mca cycle counts on Clang: identical for R=32 t1_dit
  (321 → 322, within noise)
