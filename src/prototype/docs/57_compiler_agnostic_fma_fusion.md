# 54. Compiler-agnostic FMA fusion — investigation and negative result

## TL;DR

A 4-compiler survey on R=64 c2c (same OCaml-generated C source, three
compilers: gcc-11, gcc-13, clang-18) revealed a ~7% performance gap on
clang-18 driven by **clang failing to fuse `Add(Mul, c)` patterns into
FMA instructions** the way gcc does, plus 2.5× more register-allocator
spills.

We hypothesized that doing controlled FMA lifting in `algsimp.ml`
would close this gap. Implementation showed:

- **Source-level**: lifted version had 235 explicit FMAs (up from 160,
  approaching gcc-11's 286 fused) and matched gcc-11's total op count
  (1305 vs 1304). Prime correctness 56/56 PASS.
- **Compilation failed**: emitted source referenced tags that were not
  declared, due to an unanticipated interaction between hashconsing,
  spill markers, and emit_c's topological walk.

The root cause is a **structural invariant about DAG reachability that
the existing fma_lift respects (by being gated to primes, where there
are no spill markers) but fma_lift_safe broke** (by running on
composites where spill markers exist as independent DAG roots).

Conclusion: the controlled-lift hypothesis is mechanically sound and
the safety conditions (single-use Mul, cross-pass skip) are correct,
but a full implementation requires a coordinated rewrite pass that
handles `assigns` + `spill_markers` (and possibly other DAG roots)
together with explicit topology-preservation invariants. That's ~week
of focused work, not hours. **For the current public release, the
clang gap is documented as a known limitation.**

## Motivation

The original framing: "improve schedule.ml ordering." Diagnostic on
gcc-11/gcc-13 showed:
- gcc-11 sched-on: 75526 cycles, 220 spills, IPC 3.35
- gcc-13 sched-on: 76330 cycles, 259 spills, IPC 3.41
- OCaml source order on gcc-11 (sched-off): 77432 cycles, IPC 3.27

A ~3% scheduling gap. Option B (port-balance tiebreaker in
schedule.ml) was implemented and measured: ±0.7% across radixes, with
crossover at R~96 — small, mixed, not worth shipping. Bigger projects
identified:

1. Cross-compiler variance (this doc)
2. Spill structure at R=128/R=256 (turned out to be a hard structural
   floor: 595 spills at R=128 is invariant under `--fuse`; gcc's RA
   produces the same total regardless of our markers)

The driving question for this investigation: **for a public code
release, what's the worst-case performance gap a user could see if
they build with a compiler other than gcc-11?**

## Methodology — compiler survey

Same source (R=64 c2c with twiddles, in-place, default scheduler),
three compilers, icelake-server target, llvm-mca measurement on full
loop body.

```
Compiler   FMA count   Spills    Cycles    IPC      vs gcc-11
gcc-11     286         220       78306     3.24     baseline
gcc-13     286         259       79204     3.30     +1.1%
clang-18   160         555       84108     3.25     +7.4%
```

Two distinct issues with clang:

1. **No mul+add fusion.** clang keeps the 160 FMAs we emit explicitly
   (from algsimp's `lift_sub_neg_mul` and other prime-shape patterns)
   but finds zero additional ones. gcc-11/gcc-13 both find 126 more
   from our `_mm512_mul_pd` + `_mm512_add_pd` source patterns.
2. **2.5× more RA spills** (555 vs 220 on gcc-11).

The 7.4% cycle gap is dominated by the second issue but the first is
real.

The point of the survey: anyone building this with a default compiler
on a modern Linux distro will likely get clang or recent gcc, not
gcc-11 specifically. The performance numbers in the README would
diverge from what users observe by up to 7%.

## Why our source doesn't have those 126 FMAs already

Doc 28 established that `algsimp.fma_lift` (which fuses ALL
`Add(Mul, c)` patterns into NK_Fma atoms) causes **33-48% regression**
on composite codelets. Two mechanisms:

1. RA constraint: explicit NK_Fma encodes operand assignment;
   gcc's pattern-matching mul+add lets gcc pick FMA variant
   (132/213/231) freely based on register pressure
2. Cross-pass interactions: NK_Fma bridges PASS 1 / PASS 2 boundaries
   differently than Add(Mul, c); classify_passes / inline_set was
   tuned for the pre-lift node shapes

So fma_lift is gated to primes (Direct construction) where it gives
1-2% benefit. For composites, we *deliberately leave the mul+add
patterns un-fused* and rely on gcc's pattern matcher.

gcc-11 and gcc-13 do this fusion well (286 FMAs each). **clang
doesn't.**

## Hypothesis — controlled lift

The full fma_lift had three behaviors that doc 28 implicated:
1. Lifts every Add(Mul, c) in the DAG (depth-unrestricted)
2. `liftable_mul = true` unconditionally — duplicates shared Muls
3. No spill-marker awareness — rewrites nodes that bridge pass boundaries

A *controlled* variant could potentially capture clang's FMA win
without triggering gcc's regression by tightening to:
1. Restricted lifts (depth or use-count based)
2. `single_use` requirement — no duplication
3. Skip spill-marker nodes

The single-use restriction is the key safety property: it means a
lifted Mul is guaranteed to become unreachable from outputs (no
duplicate computation needed). Each lift preserves total op count
(1 mul + 1 add → 1 fma); no duplication, no RA pressure added.

## Implementation attempts

### Attempt 1: `fma_lift_sinks` (depth-0)

Walk only the top-level expression of each output assignment; lift
only if outermost is `Add(Mul, c)` / `Sub(Mul, c)` pattern with
single-use Mul.

**Result at R=64**: 0 lifts of 128 outputs scanned.

Diagnostic showed top-level operand kinds:

```
Sub(Sub[uses=2], Sub[uses=2]) = 32 outputs
Add(Sub[2], Sub[2])           = 32
Add(Add[2], Add[2])           = 24
Sub(Add[2], Add[2])           = 24
Sub(Add[2], Fma[2])           = 8
Add(Fma[2], Add[2])           = 8
```

**Structural finding**: composite CT codelet outputs are always
sums-of-butterfly-outputs. The Mul-of-twiddle lives in the butterfly
intermediates, never at the output sink layer. The depth-0 rule was
designed for prime DAGs (Direct construction) where Mul *does* appear
at the sink. Doesn't apply to composites.

For primes, fma_lift already runs (aggressive=true gate). So
fma_lift_sinks would be a no-op everywhere it could help.

### Attempt 2: `fma_lift_sinks` extended to depth-1

Process top-level Add/Sub plus its direct operands. Cache shared
intermediates so each one gets a consistent lift decision.

**Result at R=64**: 8 lifts (160 → 168 source FMAs). Slightly better,
but the vast majority of fusable patterns are at depth ≥ 2.

### Attempt 3: `fma_lift_safe` (full DAG walk, single-use only)

Walk the full DAG like the existing fma_lift, but with two
restrictions:
- `single_use m_node` for the candidate Mul (no duplication)
- `is_spilled n` skip for spill-marker nodes

**Source-level result**: 235 FMAs (close to gcc-11's 286), total arith
ops 1305 (matching gcc-11's 1304). Op-count metrics looked exactly
right.

**Correctness**: prime test suite 56/56 PASS.

**Compilation**: failed. Multiple undefined-tag errors:

```
const __m512d t2164 = _mm512_fmsub_pd(t570, t572, _mm512_mul_pd(t573, t2162));
                                                                       ^
                                                  use of undeclared identifier 't2162'
```

Despite the correctness suite passing (it doesn't exercise composites
with spill), the emitted source had references to tags that emit_c
didn't declare.

## Root cause — three DAG roots and hashcons

The bug isn't in the lift logic; it's in a tacit invariant about DAG
reachability that I didn't recognize.

Three independent "DAG roots" reference algsimp tags:

1. **assigns** — the output store assignments `(elem_ref, t) list`,
   passed through fma_lift_safe and then to emit_c
2. **spill_markers** — `spill_tag_marker list` with `re_tag` and
   `im_tag` fields naming tags that PASS 1 stores and PASS 2 reloads
3. (Possibly other paths through the IR that I haven't audited)

The existing fma_lift respects these implicitly: it only runs for
aggressive=true (primes), where `spill_markers = []`. There's only one
DAG root in that case (assigns), and rewriting it is straightforward.

For composites, spill_markers is non-empty. My fma_lift_safe walks
`assigns`, creates new nodes via `mk_mul`, `mk_add`, `hashcons`, and
**hashconsing can return existing nodes** — including nodes that are
only reachable through spill_markers, not through assigns.

The failure mode:
1. fma_lift_safe rewrites some Add(Mul, c) into Fma(a, b, c) via
   `hashcons (NK_Fma (a, b, other, false, false))`
2. The `c` operand is the rewritten `b'` from elsewhere in the tree
3. `b'` references a Mul whose `b` operand was a spill marker subtree
   value
4. Hashcons resolves the inner Mul to an existing node — one reachable
   only through spill_markers
5. emit_c walks assigns, follows pointers to the new Fma, then to its
   operands, then to inner Muls
6. emit_c reaches the existing Mul node whose subtree references a
   spill-marker-only tag (`t2162`)
7. emit_c emits the reference as `t2162` but never declared it (its
   topological walk didn't visit `t2162` separately)

The `is_spilled` check I added covers case (3) — don't lift nodes that
ARE spill markers themselves — but doesn't cover this transitive case
where a rewritten node *reaches* a spill-marker subtree via hashcons
unification.

## What a real fix would require

A correct controlled-lift implementation needs the rewrite to:

1. **Enumerate all DAG roots before rewriting** — assigns AND
   spill_markers (re_tag and im_tag), not just assigns.
2. **Build a unified reachability set** across all roots before
   computing use_count. Single-use tests must consider all reachable
   uses, including spill_marker references.
3. **Rewrite all roots through the same cache** so that tag references
   from spill_markers resolve to rewritten nodes consistently.
4. **Propagate cache results back to spill_markers**: if a spill
   marker's `re_tag` got rewritten, the marker needs updating to point
   to the new tag (or the rewrite must be prohibited for that tag).

Option (4) split: either we don't rewrite spilled tags at all (which
is what `is_spilled n` was meant to do, but it only catches direct
spill markers, not transitive subtree references), or we rewrite them
and update spill_markers.

The simpler version of (4) — *never rewrite anything reachable from
spill_markers* — is the conservative path. It would limit lifts to
purely-output-reachable patterns, which at R=64 is a smaller set than
the 235 we achieved (but presumably nonzero).

Either way, this is ~week of careful work touching algsimp.ml,
gen_radix.ml's lift_spill_markers pipeline, and possibly emit_c.ml.
Significant chance of breaking other invariants in the process.

## What ships in the current release

| Compiler | R=64 c2c cycles | vs gcc-11 |
|---|---|---|
| gcc-11   | 78306 | baseline |
| gcc-13   | 79204 | +1.1% |
| clang-18 | 84108 | +7.4% |

The clang gap is real and known. It's primarily two effects:

1. **No mul+add fusion**: gcc's pattern matcher reliably finds
   `Add(_mm512_mul_pd(...), c)` and emits FMAs; clang doesn't. We
   emit 160 explicit FMAs from algsimp's existing prime-shape lifts
   (`lift_sub_neg_mul` and Cmul construction); gcc finds 126 more,
   clang finds 0 more.
2. **RA spill density**: clang's register allocator spills 2.5× more
   than gcc-11 on the same R=64 source (555 vs 220). This is a
   property of clang's RA at this codelet size; not addressable from
   our IR side.

The 7.4% gap is dominated by spill density. Even closing the FMA gap
fully would leave most of the difference unaddressed.

## What's worth keeping

This investigation produced several pieces worth preserving in the
public-release narrative:

1. **The 3-compiler survey methodology**. Same source, three
   compilers, isolate the per-compiler delta. Reusable for any future
   release-verification work.

2. **The structural finding about composite CT output shapes.**
   Outputs are sums-of-butterfly-outputs; the Mul-of-twiddle lives in
   intermediates, not sinks. This contradicts the prime-DAG intuition
   and explains why fma_lift helps primes (1-2%) but is gated for
   composites.

3. **The three-roots invariant.** The IR has multiple independent DAG
   roots (assigns + spill_markers + possibly more); hashcons unifies
   nodes across them; rewrite passes must coordinate all roots or
   they'll break invisible invariants. This is the kind of thing
   that's only obvious after you've hit it.

4. **The negative result on fma_lift_safe.** Saved bytes of
   well-intentioned but broken code; documented why a "simple"
   improvement isn't.

## Files touched

- `lib/algsimp.ml` — added `fma_lift_safe` (currently unused; left in
  source as documented attempt with clear caveats above the function)
- `bin/gen_radix.ml` — reverted to original `if aggressive then
  fma_lift post_trans else post_trans` pattern
- `lib/schedule.ml` — Option B port-balance code remains (active for
  all radixes; net effect ±0.7%, crossover at R~96 favoring large
  sizes). Could be flagged behind `--port-balance` if cleaner is
  desired.

No production behavior changes from this investigation.

## Open questions for future work

1. **Is clang's spill density addressable at all from our IR side?**
   Reducing live ranges or providing register hints might help, but
   we have no evidence this works. Would require empirical
   investigation.

2. **Would gcc-15 or newer clang versions fix the FMA fusion gap on
   their own?** Compiler quality moves; today's variance might shrink
   in 2 years. Worth re-running the survey on each major release.

3. **The `--fuse` parameter is currently a no-op for performance**
   (doc this session showed total spill count invariant across
   fuse ∈ {0, 2, 4, 6, 8} on both gcc-11 and gcc-13). Consider
   removing or gating to a documented experimental flag.

4. **The Option B port-balance code is currently active in
   schedule.ml** with mixed-direction effect (regression at R=32/64,
   marginal win at R=128/256). Decision: ship as-is, gate behind flag,
   or remove. Recommendation: gate behind `--port-balance` flag for
   clarity, since the small net effect isn't worth defaulting.

## Status

- ✓ 4-compiler survey methodology established; data preserved
- ✓ Clang 7.4% gap mechanism identified (no mul+add fusion + 2.5× RA spills)
- ✓ Three failed controlled-lift attempts measured and documented
- ✓ Root cause identified (three-DAG-roots invariant + hashcons unification)
- ✓ Working baseline restored; 56/56 prime correctness intact
- → Full fix deferred (~week of careful work, non-trivial regression risk)
- → Clang gap shipped as documented limitation in public release
