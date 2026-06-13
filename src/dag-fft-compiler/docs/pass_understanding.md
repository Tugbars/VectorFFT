# What each optimization pass actually solves

For each item: the **problem** it addresses, the **mechanism**, **when it
applies** vs not, and **what breaks** if you omit or misapply it.

Scope: I focus on what matters for our Bailey radices (R=8/16/32/64,
CT-decomposed, n1 and t1). Passes that only fire in `aggressive` mode
(Direct primes R=3/5/7/11/13/17 — Winograd structure) are noted as
non-applicable for our use case but documented for completeness.

═══════════════════════════════════════════════════════════════════════
PASS 1: `of_assignments ~reassoc`
═══════════════════════════════════════════════════════════════════════

**Problem**: The math layer emits Expr.expr (concrete tree). We need
hash-consed Algsimp.t (sharing-aware DAG) so identical subexpressions
are unified into one node. Without this, the same value gets computed
multiple times.

**Mechanism**: Walk each output's expression tree. For each subexpression,
look it up in the hash-cons table; if found, return the existing node
(shared); else create one. Also runs smart constructors (`mk_add`,
`mk_sub`, `mk_mul`) which apply peephole rewrites at construction time:
- `Add(x, Neg(y)) → Sub(x, y)`
- `Sub(Neg(Mul(a,b)), c) → Fma(a, b, c, true, true)`  (the `lift_sub_neg_mul` peephole)
- Constant folding for Const × Const, Const + Const

`reassoc=true` (the default for CT codelets) also tries reassociation
during construction. The flag is per-codelet: `Dft.needs_reassoc n`
decides.

**When applies**: always. Every codelet path runs this.

**If omitted**: catastrophic — without hash-consing, the DAG would be
exponentially redundant. We already do this in codelet_oop.

═══════════════════════════════════════════════════════════════════════
PASS 2: `dedup_sub_pairs`
═══════════════════════════════════════════════════════════════════════

**Problem**: After reassociation, both `Sub(a,b)` and `Sub(b,a)` may
appear as separate nodes despite being negatives of each other. They
were built independently from different paths in the DAG, so
hash-consing didn't unify them.

**Mechanism**: Build (a.tag, b.tag) → Sub-node index. For each pair
where both `(a,b)` and `(b,a)` exist, pick a winner (higher usage count;
lower tag if tied) and rewrite uses of the loser as `Neg(winner)`. The
smart constructors' `Add(x, Neg(y)) → Sub(x, y)` peephole then collapses
the result during rebuild.

**When applies**: any DAG where reassociation creates such pairs. In
practice: most CT codelets at R ≥ 8.

**If omitted**: ~3-8% redundant ops at R=16, more at higher R. Both
`Sub(a,b)` and `Sub(b,a)` get materialized as separate registers, doubled
arithmetic for what's algebraically one value. Hits FMA fusion too —
the duplicate Sub may consume a Mul that, if shared properly, would
fuse into an FMA at exactly one site.

**Misapply risk**: low — the pass is conservative (only rewrites when
both directions exist).

═══════════════════════════════════════════════════════════════════════
PASS 3: `factor_common_muls` (aggressive mode only — Direct primes)
═══════════════════════════════════════════════════════════════════════

**Problem**: For monolithic prime DFTs (R=3/5/7/11/13/17 etc.), the
direct DFT formula produces `Σ ± c_j · x_j` per output. The Winograd
structure (sums and differences of pairs at conjugate frequencies)
emerges when we group same-`c_j` terms across signs: `c·x_a + c·x_b →
c·(x_a + x_b)`.

**Mechanism**: Flatten Add/Sub chains into signed term lists. Group by
the constant coefficient. Build `c · Σ(±x_i)` for each group. Operates
on flat sums (not binary pairs) because the binary form orders by tag,
which scatters same-c terms.

**When applies**: ONLY for monolithic primes (`--aggressive` flag set
by `pick_algorithm n = Direct`). CT-decomposed codelets explicitly skip
this — their twiddle muls have distinct constants per output, so no
grouping opportunity.

**Misapply risk**: HIGH if run on CT codelets. The docstring spells it
out: "in CT-decomposed codelets the same Mul(xr, k) is shared between
Cmul Re and Im outputs (use_count ≥ 2). Factoring naively would destroy
that sharing." Empirically: R=16 with full aggressive on CT regressed
+94 ops. **This is exactly the kind of mistake we'd make without
reading the code.**

**For us (R=8/16/32/64 CT)**: pass through unchanged. The pass is a
no-op when `aggressive=false`. We must NOT set aggressive=true.

═══════════════════════════════════════════════════════════════════════
PASS 4: `factor_by_atom` (aggressive mode only — Direct primes)
═══════════════════════════════════════════════════════════════════════

**Problem**: Complementary to factor_common_muls. Groups by the
NON-constant operand: `c1·x + c2·x + c3·x → (c1+c2+c3)·x`. Since c1, c2,
c3 are compile-time Consts, the inner sum folds to one Const at hash-cons
time → N muls collapse to 1.

**Mechanism**: For each Add chain, extract (coefficient, atom) pairs.
Hash by atom. Sum coefficients. Emit `Mul(folded_const, atom)` per atom.

**When applies**: same as factor_common_muls — primes only.

**For us**: no-op. Don't enable aggressive.

═══════════════════════════════════════════════════════════════════════
PASS 5: `share_subsums` (aggressive mode only — Direct primes)
═══════════════════════════════════════════════════════════════════════

**Problem**: After factoring fires on primes, the DAG contains pair
sums like `s14 = x[1] + x[4]` (built for `0.309·s14`). The X[0] output
is `x[0] + x[1] + x[2] + x[3] + x[4]` (5-term sum) — could be expressed
as `x[0] + s14 + s23`, saving 2 ops per X[0] output.

**Mechanism**: For each Add chain, look for pre-existing pair-sums in
the hash-cons table that match two terms; substitute them.

**For us**: no-op. Don't enable aggressive.

═══════════════════════════════════════════════════════════════════════
PASS 6: `collect_m` (opt-in via VFFT_COLLECT_M, default off)
═══════════════════════════════════════════════════════════════════════

**Problem**: Like factor_by_atom but for any Add/Sub subtree, not just
prime-driven ones. `ax + bx + cx → (a+b+c)·x`, `ax - bx → (a-b)·x`,
`ax + x → (a+1)·x`. The Const-fold of (a+b+c) collapses muls.

**Mechanism**: For each Add/Sub flat sum, extract (coeff, atom) per
term, group by atom tag, sum coefficients (compile-time fold), emit
new Mul(c_folded, atom). Cache results by tag to avoid redundant work.

**When applies**: opt-in. Default OFF. Production codelets don't enable
it; it's an experimental optimization.

**For us**: don't enable. Matches production.

═══════════════════════════════════════════════════════════════════════
PASS 7: `deep_collect` (opt-in via VFFT_DEEP_COLLECT, default off)
═══════════════════════════════════════════════════════════════════════

Like collect_m but recursively distributes Const·(Add/Sub) through
inner subtrees to expose more atoms. Default OFF. Skip.

═══════════════════════════════════════════════════════════════════════
PASS 8: `Algsimp.transpose` + iterated factor loop (aggressive only)
═══════════════════════════════════════════════════════════════════════

**Problem**: Frigo's network transposition trick — for linear DAGs,
reversing edge direction and swapping (in,out) is mathematically
equivalent but produces a different op count after subsequent
factoring. Iterate: factor → share → transpose → factor → share → ...
until op count stops decreasing.

**When applies**: aggressive ∧ no Cmul nodes ∧ not direct-prime.
For us: gen_radix sets `aggressive ↔ pick_algorithm = Direct`, and
direct primes also disable transposition (the inner `not is_direct`
guard). So in practice transpose is unused for our use case.

**For us**: skip.

═══════════════════════════════════════════════════════════════════════
PASS 9: `fma_lift` — single-use FMA absorption ★
═══════════════════════════════════════════════════════════════════════

**Problem**: gcc auto-fusion of `Mul + Add` into FMA is unreliable —
especially when emission inserts fences (`asm volatile`) between the
mul and the add. The IR has separate NK_Mul and NK_Add nodes; without
explicit fusion, the emitter produces:
```
register __m512d t1 = _mm512_mul_pd(a, b); asm volatile ("" : "+v"(t1));
register __m512d t2 = _mm512_add_pd(t1, c); asm volatile ("" : "+v"(t2));
```
gcc cannot fuse across the asm volatile barrier on t1 → emits separate
vmulpd and vaddpd instead of one vfmadd. **This is exactly the 22
standalone muls we see in our OOP codelets.**

**Mechanism**: Recognize patterns
- `Add(Mul(a,b), c)` where Mul has use_count = 1 → `Fma(a, b, c, F, F)`
- `Sub(Mul(a,b), c)` → `Fma(a, b, c, F, T)` (= vfmsub)
- `Sub(c, Mul(a,b))` → `Fma(a, b, c, T, F)` (= vfnmadd)
- Same with `Neg(Mul(a,b))` instead of `Mul(a,b)` → vfnmadd/vfnmsub
- `Sub(Neg(Mul(a,b)), c) → Fma(a, b, c, T, T)` (= vfnmsub) — handled
  via the `lift_sub_neg_mul` peephole at construction time

After fma_lift, NK_Fma is OPAQUE to all downstream passes (none of them
recursively rewrite into Fma operands).

**Critical constraint**: only lift when `use_count(Mul) = 1`. If the
Mul has multiple consumers, naive lifting would either DUPLICATE the
Mul across each consumer (more ops, not fewer) or BREAK SHARING. fma_lift
refuses these cases.

**Frozen tags**: when spill markers are active, the spill targets'
tags must NOT be rewritten — `emit_c` walks only the reachable subset
from `assigns`, and spilled values are referenced by tag only. If
fma_lift rewrites a spill target into an Fma, the spill markers point
to dead nodes. Solution: pass `frozen_tags` set to fma_lift; nodes in
the set are returned unchanged.

**When applies**: CT-decomposed codelets (Cooley_Tukey case in
`pick_algorithm`). Not Split_radix (uncalibrated; would regress).

**Misapply risk**: HIGH if frozen_tags not threaded through. Would
silently corrupt the DAG when spill is active. R=32/64 t1 codelets
specifically need this protection.

**Empirical impact** (per doc 56): R=32 t1 SU+spill went from 33-48%
regression (with old broken policy) to 5.2% WIN with current single_use
policy. With M-project regalloc combined: 26.8% win.

═══════════════════════════════════════════════════════════════════════
PASS 10: `factor_const_muls` — multi-use factoring before FMA absorption
═══════════════════════════════════════════════════════════════════════

**Problem**: fma_lift refuses to absorb Muls with use_count > 1
(because duplication = wrong). But many CT codelets have patterns like:
`Add(Mul(K, X), Mul(K, Y))` where K is the same Const (a twiddle real
or imag part). Both Muls have use_count = 1 individually, but together
they share the same K-multiplication structurally.

**Mechanism**: Recognize `Add(Mul(K, X), Mul(K, Y)) → Mul(K, Add(X, Y))`
(and Sub variant). The resulting outer Mul has the SAME use_count as
the original Add (it inherits both consumers), AND the inner Add is
shared if X+Y or X-Y already existed.

After this, the outer `Mul(K, sum)` is a single multiplication; if its
consumer is itself an Add/Sub with a third operand, the next pass
(multi_use_fma_lift) absorbs it.

**Frozen tags**: must be threaded. Returns a `remap` table; caller must
`extend_frozen` the remap (so spill markers tracking these tags follow
the rewrite). This is the chain `factor_tag_remap → mfl_tag_remap →
fma_addend_remap → ...` in gen_radix.

**When applies**: only when fma_lift is enabled (i.e., CT codelets).
Conservatively gated: doesn't fire if it would touch frozen subtrees
or destroy shared partial-sums.

**If omitted**: leaves standalone Muls in patterns where they could
have been collapsed before FMA absorption. Op count higher than FFTW
hand-coded equivalent.

═══════════════════════════════════════════════════════════════════════
PASS 11: `multi_use_fma_lift` — absorbs Muls into ALL their consumers
═══════════════════════════════════════════════════════════════════════

**Problem**: fma_lift only absorbs Muls with use_count = 1. But after
factor_const_muls produces an outer `Mul(K, sum)` shared by multiple
consumers, that Mul still has use_count > 1. Pure fma_lift refuses.

**Mechanism**: For Muls whose use_count > 1 BUT EVERY consumer is an
Add/Sub that can absorb the Mul as an Fma, rewrite each consumer into
its own Fma. The Mul becomes dead (zero remaining consumers). Net: same
op count if N consumers each become Fmas, no Mul. Saves N-1 ops if the
Mul was duplicated, but the real win is enabling FMA at every consumer
site.

**Frozen tags**: same threading as factor_const_muls.

**Iterated**: gen_radix runs this 4 times alternating with
fma_addend_factor, because each pass may create new Mul patterns that
the next can absorb. Diminishing returns; runs ~3 useful iterations
typically.

**If omitted**: ~5-10% more ops than FFTW hand at R=8/16/32 because
post-factoring Muls don't get absorbed.

═══════════════════════════════════════════════════════════════════════
PASS 12: `fma_addend_factor` — factor Fma's addend slot
═══════════════════════════════════════════════════════════════════════

**Problem**: After previous passes we have `Fma(K, X, Mul(K, Y), nm, na)`
patterns where the FMA's multiplicand slot has constant K and its
addend slot is `Mul(K, Y)` with the SAME K. Vanilla factor_const_muls
only sees Add/Sub patterns; it misses this Fma+Mul structure.

**Mechanism**: Refactor to `Mul(K, X±Y)` so K-multiplication becomes a
single outer Mul on the (X±Y) sum/diff. Follow-up multi_use_fma_lift
then absorbs the outer Mul into downstream consumers.

**Empirical impact** (per gen_radix comment): "Closes R=8/16 to FFTW
exactly and saves ops at all larger radices."

═══════════════════════════════════════════════════════════════════════
PASS 13: `flatten_fma_mul_addend` — 2-FMA chain from Fma+Mul
═══════════════════════════════════════════════════════════════════════

**Problem**: Residual `Fma(A, B, Mul(C, D), nm, na)` patterns where the
addend Mul couldn't be factored (different constants). Still leaves a
standalone Mul.

**Mechanism**: When this Fma feeds into an Add/Sub with a third operand
P, rewrite to a 2-FMA chain: `Fma(C, D, Fma(A, B, P, _, _), _, _)`.
Eliminates the Mul. Saves 1 op per occurrence.

This is the "Cat-B finisher" per doc 59.

═══════════════════════════════════════════════════════════════════════
PASS 14: `lift_spill_markers` — Expr.expr tags → Algsimp.t tags
═══════════════════════════════════════════════════════════════════════

**Problem**: Spill markers from dft_expand_*_blocked/*_spill carry
Expr.expr subtrees, not Algsimp.t tags. We need the Algsimp tag of
the hash-consed version of those subtrees so make_spill_info can index
by tag.

**Mechanism**: For each marker, call `of_expr ~reassoc` on the re_expr
and im_expr. Returns the Algsimp.t with its tag. Build new marker list
with tags.

**Frozen tags chain**: after fma_lift + the cascade, the tag may have
been rewritten. Chain remap tables (factor_tag_remap → mfl_tag_remap →
... → flatten_tag_remap) and apply them in order to find the FINAL
tag. This is what `remap_tag` in gen_radix does.

═══════════════════════════════════════════════════════════════════════
PASS 15: `make_spill_info` with ?ct ?fuse
═══════════════════════════════════════════════════════════════════════

**Problem**: emit_c needs a structured table mapping (tag → slot) for
spill targets, plus the (n1, n2) CT factorization so it can cluster
PASS 2 sub-FFTs and identify fused slots.

**Mechanism**: Build re_slot and im_slot hash tables from markers.
With `?ct=(n1,n2) ?fuse=M`: mark the last M sub-DFT-n2 output positions
in each PASS 1 sub-FFT as "fused" — these values are KEPT IN REGISTERS
across the PASS 1/PASS 2 boundary instead of being stored to spill_re/
spill_im and reloaded.

**Why fuse**: PASS 1 emits values in some order; the LAST values emitted
are the FIRST consumed by PASS 2 (in CT structure, sub-DFT-n1 #k2
consumes slots {n1_idx*n2 + k2 : n1_idx in 0..n1-1}). Keeping a few of
these in registers across the pass boundary saves load+store traffic
for them.

═══════════════════════════════════════════════════════════════════════
PASS 16: `classify_passes` — PASS 1 / PASS 2 split
═══════════════════════════════════════════════════════════════════════

**Problem**: With spill_info, the codelet body emits in two passes
with explicit stores/loads at the boundary. Which nodes go in PASS 1
(computed and stored) vs PASS 2 (loaded and consumed)?

**Mechanism**: A node is PASS 2 iff it transitively depends on a
spilled tag. Otherwise PASS 1. The spilled tags themselves are PASS 1
(they're the boundary — computed in PASS 1, stored, reloaded in PASS 2).

Walk in topological order; for each node, classify based on whether
any pred is PASS 2 (or the tag itself is spilled).

**Why this matters**: bounds peak_live per pass. PASS 1's live set is
the sub-DFT-n2 working values plus spill destinations. PASS 2's live
set is the loaded spill values plus sub-DFT-n1 working values. Each
pass individually fits in 32 zmm.

═══════════════════════════════════════════════════════════════════════
PASS 17: `compute_inline_set` — single-use inlining
═══════════════════════════════════════════════════════════════════════

**Problem**: Each named `const __m512d tN = ...;` declaration adds an
SSA value gcc's allocator must track. Hand-coded FFTW codelets nest
intrinsics: `const __m512d t2 = _mm512_mul_pd(K, _mm512_sub_pd(a, b));`
— our linearized form: `t1 = sub; t2 = mul(K, t1);`. The hand-coded
form has fewer named values → fewer SSA bindings → lower register
pressure.

**Mechanism**: Count uses per tag. For tags with use_count = 1 (and
not Load, Cmul, or output sink), mark them in inline_set. When
render_node_def emits a consumer that references an inline_set tag,
it inlines the expression instead of emitting `t<tag>`.

**Empirical impact** (per docstring): closes the gap to hand parity
on R=11/13/17 t1_dif (and helps DIT too).

**For us**: this is one of the cheapest, most universal wins. Should
be on for everything.

═══════════════════════════════════════════════════════════════════════
PASS 18: Schedulers — Topological vs Bisection vs SU vs BB
═══════════════════════════════════════════════════════════════════════

**Topological** (by tag): simplest. Tags are assigned in hash-cons
order, which is bottom-up construction order. Sort ascending = a valid
topological order. No latency-awareness, no register-pressure
awareness.

**Bisection** (Frigo's): recursive scheduling that bisects the DAG at
midpoints. Produces a Seq tree. Works with Annotate. Used for n1
monolithic codelets traditionally.

**SU (Sethi-Ullman)**: list scheduler with priority = (cp_dist DESC,
su_number ASC).
- `cp_dist` = critical-path distance to sink in cycles (weighted by
  uarch latencies). High cp_dist → schedule early to expose ILP.
- `su_number` = approximate register pressure to evaluate this subtree.
  Low su_number → schedule first (start with subtrees that hold few
  regs simultaneously).

For our purposes: SU is more effective than Topological at exposing
ILP and bounding peak_live. Empirically ~5-15% over Topological at R≥32.

**BB (branch-and-bound)**: lexicographic cost (saturated_peak, -progress).
Searches the space of schedules. Default off; opt-in via --bb. Used at
R=64 AVX2 K=512-1024 (+5.8% over SU+GH per doc 22).

**Goodman-Hsu (GH) mode**: AVX2-specific tweak for R≥32; auto-enabled
when su && vec_regs ≤ 16 && n ≥ 32.

═══════════════════════════════════════════════════════════════════════
PASS 19: `Schedule.su_schedule_subset` — cluster-local scheduling
═══════════════════════════════════════════════════════════════════════

**Problem**: With spill structure, PASS 1 has clusters (one per sub-DFT-
n2). Within a cluster, nodes form a sub-DFT computation; across clusters,
sub-DFTs are independent. Scheduling globally with SU might interleave
sub-DFTs, but that hurts cache/register locality.

**Mechanism**: Group pass1_blocked_topo by cluster (`min_slot / n2`).
Within each cluster, run su_schedule_subset with the cluster's nodes
and the cluster's spill sinks. Output the per-cluster schedules
concatenated.

**Empirical**: cluster-local scheduling preserves PASS 1 sub-DFT structure
that monolithic SU would scramble. Combined with spill, this is the
recipe.

═══════════════════════════════════════════════════════════════════════
PASS 20: `Regalloc.allocate` — SSA linear-scan with chordal coloring
═══════════════════════════════════════════════════════════════════════

**Problem**: gcc's register allocator is heuristic and doesn't always
produce the optimal assignment for codelet-style straight-line code with
many short-lived intermediates. For log3 AVX-512 R≤32, the gen_radix
codelet has enough structure that an SSA chordal-coloring allocator
beats gcc's heuristics.

**Mechanism**: SSA-form code is chordal (live ranges form interval
graphs). Chordal graphs can be optimally colored in polynomial time.
Linear-scan over the scheduled list assigning physical registers
(zmm0..zmm31). When pressure exceeds budget, allocate spill slots
(num_spill_slots in regalloc_spill[N]).

**Output**: an `allocation` record with assignment table (tag → Reg
"zmmK" | Spilled slot | Default), spill_sites map (position → spills),
reload_sites map (position → reloads), num_spill_slots.

**Two-rule policy says: turn ON only for log3 AVX-512 R≤32**. For other
cases, fall back to fence-only emission (let gcc allocate). The empirical
basis (per docs/fence_pin_decomposition.md): the fence is the primary
win mechanism; the pin is a narrow-band benefit (log3 only) and an
active cost in most other cases.

**For us (Bailey n1/t1 at R=8..64)**: regalloc is OFF by default. We use
fence-only emission instead.

═══════════════════════════════════════════════════════════════════════
PASS 21: Fence emission (`current_fence_only := true`)
═══════════════════════════════════════════════════════════════════════

**Problem**: gcc's instruction scheduler is free to reorder
instructions across our SU+GH ordering. SU+GH carefully placed nodes
to bound register pressure and expose ILP; gcc reordering can defeat
this. Empirically (per docs/fence_pin_decomposition.md), gcc's
"improvement" is often worse than the SU+GH schedule.

**Mechanism**: After each `register __m512d tN = expr;`, emit
`asm volatile ("" : "+v"(tN));`. This is a side-effect barrier that
constrains gcc's scheduler to honor the source order around tN's
definition. tN's value goes through an opaque asm operation; gcc
cannot move computations across the barrier.

**Critical**: the `register` keyword does NOT pin tN to a specific
zmm — it just hints that gcc should keep it in a register (gcc picks
which). The `asm("zmmK")` clause (the PIN, not the FENCE) is what
forces a specific zmm.

**When applies**: ON by default for everything EXCEPT n1 AVX2 R∈{8,16}.
That exception is empirical — those tiny codelets benefit from gcc's
freedom because the body fits in 16 ymm registers comfortably.

**Empirical**: this is THE main per-codelet win mechanism. Per docs:
"the inline-asm scheduling fence — not the register-pin clause — is the
actual win mechanism in nearly all codelets."

**For our OOP codelets**: this is the cheapest, highest-impact wiring.
Set `current_fence_only := true` before body emission.

═══════════════════════════════════════════════════════════════════════
PASS 22: Selective unpin (`compute_unpin_candidates`)
═══════════════════════════════════════════════════════════════════════

**Problem**: When regalloc IS active and pins every value with
`asm("zmmK")`, gcc cannot auto-fuse `Mul → Add → vfmadd` across the
asm volatile barrier (the Mul's value is fenced). We lose FMAs that
gcc would otherwise emit.

**Mechanism**: For NK_Mul nodes whose consumers include at least one
Add/Sub, drop the pin. Let gcc see the Mul → Add pattern and emit
`vfmaddPHS_pd` directly to the Add's pinned destination. The Mul value
disappears entirely; no intermediate register needed.

**When applies**: only when regalloc is on (log3 AVX-512 R≤32). Otherwise
the question doesn't arise (no pins to remove).

**For us**: irrelevant unless we turn on regalloc.

═══════════════════════════════════════════════════════════════════════
PASS 23: Annotate (nested scopes)
═══════════════════════════════════════════════════════════════════════

**Problem**: All `const __m512d tN = ...;` decls at function scope make
every variable appear live until function end (syntactically). gcc must
do its own liveness analysis to figure out when registers are free.

**Mechanism**: Recursively bisect the scheduled list. Decls of values
used only within a sub-block move to that block's scope. The emitted
C becomes nested `{ ... }` blocks; gcc sees scope-end as "this register
is free now."

**When applies**: opt-in via --annotate. Default off. The fence/pin
machinery achieves similar effect through different mechanism (asm
volatile pins + chordal regalloc).

**For us**: skip. Annotate and fence+regalloc are alternate approaches;
production uses fence+regalloc.

═══════════════════════════════════════════════════════════════════════
PASS 24: `dft_expand_n1_blocked` and `dft_expand_twiddled_spill`
═══════════════════════════════════════════════════════════════════════

**Problem**: Monolithic DAG construction for R≥25 produces peak_live
260-994 (8-31× over 32-zmm budget). Even with fence emission, gcc can't
work around register-budget overflow this large — it spills to stack
massively.

**Mechanism**: At the MATH layer (not the IR layer), manually unroll
the outermost CT step. For CT(n1, n2) at the top level:
- PASS 1: compute n1 independent size-n2 DFTs. Each produces n2 output
  values.
- Emit spill markers: one per (n1_idx, k2) output → slot n1_idx*n2 + k2.
- PASS 2: for each k2, compute size-n1 DFT on the loaded {slot
  n1_idx*n2 + k2 : n1_idx in 0..n1-1}.

Inner sub-DFTs (below the outermost level) still recurse via plain
`dft` — no nesting of spill structure.

The result: assignments + spill_markers + ct=(n1, n2). emit_c uses the
markers to drive PASS 1 / PASS 2 emission with explicit spill stores/
reloads at the boundary.

**When applies**:
- `should_block_n1 n vec_regs` = `pick_algorithm n = Cooley_Tukey ∧ n ≥ 25`
- `should_spill n vec_regs` = `n + 6 > vec_regs ∨ vec_regs ≥ 32 ∨ n ≥ 5`

R=25 monolithic vs blocked: 47% AVX-512 speedup, 39% AVX2.

**For us**: critical for R≥32. Without it, R=32/64 OOP codelets cannot
match production performance no matter what other optimizations we
apply.

═══════════════════════════════════════════════════════════════════════
PASS 25: Cluster-boundary store flush + fused-slot retention
═══════════════════════════════════════════════════════════════════════

**Problem**: With cluster-local SU + spill structure, when one cluster
(sub-DFT-n2) finishes, its output values are live until they get spilled
to memory. If we don't flush stores immediately at cluster boundaries,
those values stay live until end-of-PASS-1, ballooning peak_live.

**Mechanism**: Track the cluster of each PASS 1 node. When emission
crosses a cluster boundary (prev cluster's last node was the previous
iteration), immediately flush stores for the just-finished cluster's
spill targets. Their registers are now free.

**Fused slots**: marked in spill_info.fused_slots. These slots are NOT
stored at cluster boundary; instead, their values stay in registers
across the PASS 1 / PASS 2 boundary. Saves load+store for the late-
produced / early-consumed values.

**For us**: only needed when we use blocked/spill construction.

═══════════════════════════════════════════════════════════════════════
SUMMARY: what each problem looks like in the C output
═══════════════════════════════════════════════════════════════════════

| Pass missing                  | Symptom in emitted C                    |
|-------------------------------|-----------------------------------------|
| dedup_sub_pairs               | Both `Sub(a,b)` and `Sub(b,a)` appear   |
| fma_lift                      | `_mm512_mul_pd` not followed by fused   |
|                               | `_mm512_fmadd_pd`                       |
| factor_const_muls + multi_use | Shared Mul(K, sum) patterns remain as   |
|                               | separate Muls                           |
| fma_addend_factor             | Standalone Mul before/after Fma         |
| flatten_fma_mul_addend        | `Fma(_, _, Mul(_, _), _, _)` patterns    |
| compute_inline_set            | Single-use intermediates emitted as     |
|                               | separate `const __m512d tN =` decls     |
| fence emission                | Plain `const __m512d` instead of        |
|                               | `register ... asm volatile`             |
| dft_expand_n1_blocked         | Monolithic DAG with peak_live > 32      |
|                               | → massive stack spills at R≥25          |
| spill_info + classify_passes  | All values live across the whole codelet|
|                               | (no pass boundary)                      |
| SU scheduler                  | Topological order — not latency-aware,  |
|                               | not pressure-aware                      |
| cluster-local scheduling      | Interleaved sub-DFTs (bad locality)     |
| selective unpin               | Pinned regalloc kills gcc FMA fusion    |
|                               | (only relevant when regalloc is on)     |

For our R=16 example earlier: 22 standalone muls = missing fma_lift +
factor_const_muls + multi_use_fma_lift (the FMA cascade). 166 vs 144
ops = missing dedup_sub_pairs and possibly fma_addend_factor.

═══════════════════════════════════════════════════════════════════════
WHAT EACH RADIUS REGIME NEEDS
═══════════════════════════════════════════════════════════════════════

**R=8/16 n1 AVX-512**: monolithic fits in registers (peak_live ~40,
mild overflow but monolithic wins per doc 58). Wiring needed:
- dedup_sub_pairs
- fma_lift cascade (fma_lift → factor_const_muls → multi_use × 4 →
  fma_addend × 3 → flatten)
- compute_inline_set
- Fence emission

**R=16 t1 AVX-512**: same as n1.

**R=25-31 (skip; we don't use)**: should_block_n1 fires.

**R=32 n1 AVX-512**: peak_live 260 (8× budget). Wiring needed:
- ALL of R≤16 list
- PLUS dft_expand_n1_blocked
- PLUS make_spill_info
- PLUS classify_passes
- PLUS SU scheduler (cluster-local)
- PLUS spill emission machinery

**R=32 t1 AVX-512**: same as n1 but with dft_expand_twiddled_spill.

**R=64 n1 AVX-512**: peak_live 267. Same wiring as R=32. Bigger codelet
but same techniques.

**AVX2 differences**:
- vec_regs = 16, half the AVX-512 budget
- should_block_n1 threshold shifts: blocking starts to help at R≥16
- Goodman-Hsu mode auto-enabled at R≥32

═══════════════════════════════════════════════════════════════════════
THINGS THAT WOULD GO SUBTLY WRONG IF MISAPPLIED
═══════════════════════════════════════════════════════════════════════

1. **Setting aggressive=true on CT codelets** → factor_common_muls
   destroys Cmul sharing, +94 ops at R=16. Silent regression unless
   you bench against op count.

2. **Not threading frozen_tags through fma_lift cascade when spill is
   active** → spill markers point to dead nodes after rewrite.
   emit_c walks only reachable; spill stores never emit; PASS 2 reloads
   garbage. Output is NUMERICALLY WRONG, hard to debug.

3. **Running fma_lift on Split_radix DAG** → uncalibrated regression
   path. Doc 56 explicitly says don't.

4. **Enabling regalloc for non-log3 codelets** → asm("zmmK") barriers
   block gcc FMA auto-fusion. 43-126 FMAs lost on R=64 AVX-512 (per
   selective-pinning docstring). Slower than fence-only.

5. **Forgetting cluster-local scheduling with spill_info** → SU
   interleaves sub-DFTs, peak_live blows up across cluster boundaries.
   Defeats the whole point of blocking.

6. **Wrong fuse parameter** → too few fused slots = extra memory traffic;
   too many fused slots = peak_live overflow at PASS 1/2 boundary.
   Production uses fuse = ? (gen_radix's --fuse, default 0 which is
   "no fusion"). We need to look up what production uses for each R.

7. **Running net_transpose on DAGs with Cmul** → transposition is
   correct only for purely linear DAGs. Cmul is nonlinear. The pass
   has the `not has_cmul` guard for a reason; replicating it carelessly
   would corrupt t1 DAGs.

8. **Not running dedup_sub_pairs after fma_lift** → newly created Fma
   nodes' addend might be Sub(a,b) where Sub(b,a) is reused elsewhere.
   The pass list in gen_radix runs dedup BEFORE fma_lift; if we run
   it after we'd be doing different work. Need to check the ordering.

9. **Setting current_fence_only := true on AVX2 R=8/16 n1** → docs
   say specifically don't. That regime benefits from gcc's freedom.

═══════════════════════════════════════════════════════════════════════
PROPOSED WIRING ORDER (revised, with risks understood)
═══════════════════════════════════════════════════════════════════════

### Tier A — small radices (R ≤ 16), low risk

Wire what production runs for R≤16 n1 / R≤16 t1:

1. dedup_sub_pairs
2. fma_lift (CT codelets, with frozen_tags=None since no spill)
3. factor_const_muls → multi_use_fma_lift → fma_addend_factor ×3 →
   multi_use_fma_lift ×4 → flatten_fma_mul_addend (the full cascade)
4. compute_inline_set, pass to render_node_def
5. Set current_fence_only := true (except AVX2 R≤16 n1)

No scheduling change, no regalloc, no spill. Should close R=16 to ~0%
gap vs strided.

Risk: low if we get the policy gates right.

### Tier B — large radices (R ≥ 32), higher complexity

Add on top of Tier A:

6. Switch to dft_expand_n1_blocked / dft_expand_twiddled_spill when
   should_block_n1 / should_spill fires.
7. Construct spill_info via make_spill_info + tag-remap chain.
8. Wire classify_passes + Pass1/Pass2 emission with spill_sites/
   reload_sites helpers.
9. Use SU scheduler with cluster-local subset.
10. Auto-enable GH for AVX2 R≥32.

This is the bulk of the perf gain at R=32/64.

Risk: medium. The frozen_tags chain is finicky; any pass that skips
extend_frozen produces numerical-wrong codelets.

### Tier C — last 5-10%

11. Regalloc.allocate + install_alloc_canonical for log3 only.
12. Selective unpin via compute_unpin_candidates.

Risk: low (narrow scope — only fires for log3 AVX-512 R≤32, which we
aren't generating yet anyway).

═══════════════════════════════════════════════════════════════════════
KEY INSIGHT
═══════════════════════════════════════════════════════════════════════

The original phrasing "the codelet bodies are unoptimized" was misleading.
It's not one thing; it's a 12-pass IR pipeline plus a 5-step spill recipe
plus per-node emission policy. Each pass has specific preconditions,
specific risk modes, and specific empirical impact. Getting them out
of order or applying them to the wrong codelet kind can be a SILENT
regression (op count up, no warning) or a SILENT correctness bug (frozen
tags wrong, output numerically wrong).

The right approach is **mirror gen_radix.ml's exact pipeline for the
matching codelet family**, not "wire in optimization passes one by
one." For CT n1/t1, the pipeline is well-defined; deviating from it
needs justification.

For codelet_oop, the cleanest move is to factor gen_radix's pipeline
into a shared function `Vfft_v2.Pipeline.prepare_ct_codelet
~radix ~isa ~twiddled ~direction ~sign` that returns
`(assigns, spill_info, scheduler_choice, inline_set)`, and have both
gen_radix's --in-place path and codelet_oop call it. That way the
pipeline is single-source-of-truth and we don't risk drift.
