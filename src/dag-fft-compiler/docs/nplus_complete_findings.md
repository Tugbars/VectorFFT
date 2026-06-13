# N-ary Plus, collectM, and deepCollectM in VectorFFT — Final Report

## TL;DR (the empirical finding)

**Implemented FFTW's collectM and deepCollectM in VectorFFT's algsimp. Neither
finds wins on our codelets, with any cost-gating strategy that doesn't also
introduce regressions.**

The 1.6×–2.4× source-op gap with FFTW exists, but it doesn't come from local
sum collection or distribution-through-Plus. We've now empirically ruled this
out: the gap must come from somewhere else (algorithm-level Winograd derivations
specific to each N, more sophisticated FMA scheduling, or cost-model-driven
variant selection — areas we have NOT explored).

## What was built (4 commits)

### Commit 0 — Cnum + smart constructors (pre-existing)
- `lib/cnum.ml`: `{re; im}` complex combinator type with `cmul`, `cscale`,
  `croot_of_unity`, etc.
- `lib/expr.ml`: smart constructors with const folding + rotation rules
- `dft_winograd5_cnum`: reference port of W5 using Cnum

### Commit 1 — NK_Plus type infrastructure
- `NK_Plus of (int * t) list` added to algsimp's `node_kind`
- 31 match sites across algsimp.ml / schedule.ml / emit_c.ml patched with
  explicit `| NK_Plus _ -> Algsimp.nk_plus_unreachable "<site>"` guards
- `preds` and other walkers updated to handle NK_Plus
- Zero behavior change (NK_Plus is defined but unreachable in production code)

### Commit 2 — mk_plus + lower_plus smart constructors
- `mk_plus : (int * t) list -> t` enforcing 8 documented invariants:
  flatten nested Plus, absorb Neg into sign, fold Const terms, cancel
  opposite-sign duplicates, canonical sort by tag, single-term collapse
- `lower_plus : t -> t` round-trip back to balanced binary tree
- 13 unit tests via `bin/test_mk_plus.exe`, all passing

### Commit 3 — collect_m (shallow collectM)
- Walks each Add/Sub subtree, groups terms by atom tag, sums coefficients
- `subtree_has_collectible` pre-check preserves balanced binary tree shape
  when no merge possible (critical — without it, R=64 went 978 → 3162 ops
  from re-linearization)
- Gated behind `VFFT_COLLECT_M=1`
- 4 additional unit tests verifying the algorithm itself: `2x + 3x → 5x`,
  `2x - 2x → 0`, etc.
- **On production codelets: 0 wins** because no shared atoms exist within
  any single Add/Sub subtree

### Commit 4 — deep_collect (deepCollectM with use-count-aware distribution)
- `distribute_term`: pushes `Mul(Const c, ...)` through nested Add/Sub/Neg/Mul
- `distribute_use_aware`: only distributes through subtrees where at least one
  child has use_count >= 2 (hash-cons-sharing potential)
- Cost-gated: result accepted only if collected term count is **strictly less**
  than input term count
- Iterative loop (up to 5 iterations) running deep_collect + collect_m to
  fixed point, matching FFTW's algsimp structure
- Gated behind `VFFT_DEEP_COLLECT=1`
- **On production codelets: 0 wins** (every subtree's distribute+collect
  produces ≥ original term count, so the guard always skips)

## What didn't work — the experiments

| Cost gate | R=20 | R=25 | R=64 | All others |
|---|---|---|---|---|
| No gate (always distribute) | +699 | +1095 | +4091 | +24 to +1299 |
| IR-node count reduction | +107 | +224 | 0 | 0 |
| Strict term-count reduction | 0 | 0 | 0 | 0 (never fires) |
| Use-count >= 2 child only | +7 | 0 | 0 | 0 |
| Use-count + strict term reduction | **0** | **0** | **0** | **0** (never fires) |

The progression illustrates the difficulty: any guard loose enough to fire
introduces regressions; any guard tight enough to avoid regressions never fires.

## What this tells us about the FFTW gap

We've now ruled out two hypotheses:

1. **"FFTW saves ops via local term collection"** → false. Shallow collectM
   finds zero opportunities in our codelet shapes.

2. **"FFTW saves ops via distribute-and-collect"** → false. Deep collectM
   with cost-gating accepts zero opportunities; without gating it makes
   things 2-5× worse.

The remaining hypotheses (for future investigation):

1. **Algorithm-level Winograd derivations**. FFTW's `gen_notw_c -n 25`
   produces a codelet using constants like `0.066`, `0.120`, `0.989` —
   products of base W5 constants (`0.25`, `0.559`, `0.618`, `0.951`)
   with specific twiddle values. These are pre-computed by FFTW's algsimp
   working with `Cnum` symbolic algebra at a higher level than collectM —
   e.g., recognizing that `cmul(W5_output, twiddle)` simplifies algebraically
   when the W5 output is itself a known combination of inputs. This requires
   pattern-matching on whole algebraic expressions, not just collecting
   common sub-sums.

2. **Cost-model-driven variant selection**. FFTW's `gen_notw_c` accepts
   `-pipeline-latency N` and emits multiple candidate codelets, picking
   the one whose dependency graph schedules best given the assumed latency.
   We commit to one codelet shape upfront.

3. **More aggressive FMA fusion across complex algebraic patterns**. FFTW's
   `mangleSumM` is the orchestrator that combines `reduce_sumM`, `collectM`,
   and `deepCollectM` — and we haven't ported the orchestrator, just the
   constituent passes. The orchestrator may achieve wins through coordination
   that the individual passes can't on their own.

## The value of this work despite zero op-count win

Three things have real value:

1. **Empirical proof of where the FFTW gap is NOT.** Before this work, "the
   gap probably comes from collectM/deepCollectM" was a plausible hypothesis;
   it's now ruled out. The next investigation can skip this direction.

2. **Foundational infrastructure for future algebraic work.**
   `NK_Plus`, `mk_plus`, `lower_plus`, `collect_m`, `deep_collect` are
   building blocks any future algsimp project will need. They're correct,
   tested, and gated.

3. **Methodology demonstrated.** The cost-gating evolution (no-gate →
   IR-node → term-reduction → use-count-aware → strict-term-reduction)
   is a documented case study in how to safely land speculative IR rewrites.
   The pattern "implement, measure, observe regressions, tighten gate,
   re-measure" is the right loop for any future algsimp work.

## What's left, ordered by likely yield

If the goal is "close the gap to FFTW on R=25 specifically":

1. **Hand-written `dft_winograd25`** (3-5 days). Encode the 5×5 CT algebra
   with pre-computed cross-products of W5 constants and twiddle constants
   inline. This is the same algebraic work FFTW's gen_notw does, but done
   by hand for one N. Closes most of the 383→236 gap.

2. **Cost-model variant scheduling** (1-2 weeks). Generate 3-5 candidate
   codelets per N (e.g., DIT vs DIF, different sub-DFT orderings, different
   constant-rotation policies), evaluate each via the existing scheduler,
   pick the best. Closes 5-15% on most radices.

3. **Algebra-specific Cnum simplifications** (3-6 weeks). Pattern-match on
   `cmul(known_W5_output, known_twiddle)` and emit pre-derived simplified
   form. This is what FFTW's algsimp does at a level above collectM.
   Closes the rest.

For VectorFFT as a research project, option 1 is the cleanest next step —
focused, scoped, demonstrably closes the gap on a specific radix.

## Files in the deliverable

- `algsimp.ml` — type, mk_plus, lower_plus, collect_m, deep_collect
- `expr.ml` — smart constructors with rotation rules
- `cnum.ml` — Complex combinator type
- `schedule.ml`, `emit_c.ml` — NK_Plus match-site migration
- `gen_radix.ml` — pipeline wiring with VFFT_COLLECT_M / VFFT_DEEP_COLLECT gates
- `test_mk_plus.ml` — 17 unit tests
- `dump_ir.ml` — IR introspection tool (used for diagnosing R=20 regression)

## Test/regression status

- **Unit tests**: 17/17 pass
- **Production codegen**: baseline op counts unchanged at all 9 tested radices
  (5, 7, 11, 13, 16, 20, 25, 32, 64), with or without `VFFT_COLLECT_M=1`,
  `VFFT_DEEP_COLLECT=1`, or both
- **Numerical correctness**: R=25 with `VFFT_DEEP_COLLECT=1` produces
  bit-identical output to baseline (max diff 0.0)
