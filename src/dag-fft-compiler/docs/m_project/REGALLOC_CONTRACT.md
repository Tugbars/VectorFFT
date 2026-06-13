# Regalloc.allocate — Input Contract (Stage 1)

This document describes the implicit input contract that `Regalloc.allocate`
currently requires, derived by reading every existing call site in
`lib/emit_c.ml` (currently two — both inside the cluster-spill recipe).

The contract is **implicit**: it isn't checked at runtime, and the function
signature doesn't enforce it. The M-project's 9/9 composite-codelet
correctness rests on the cluster-spill recipe happening to construct inputs
that satisfy the contract. M7 (the prime/n1 extension) exposed the implicit
nature by constructing inputs that violated the contract; the result was
silent use-after-clobber bugs in R=3 AVX-512 and beyond.

This document is the prerequisite for Stage 2 (encoding the contract as
runtime assertions). Each numbered invariant below must be expressible as
a check at the top of `Regalloc.allocate`.

---

## Function signature

```ocaml
val allocate :
  isa:Isa.t ->
  scheduled:Algsimp.t list ->
  ?budget:int ->
  ?skip_tags:(int, unit) Hashtbl.t option ->
  ?inline_set:(int, unit) Hashtbl.t option ->
  ?force_last_use:(int, int) Hashtbl.t option ->
  unit -> alloc_result
```

(`?budget` defaults to `vec_regs - 4` for AVX-512, `vec_regs - 2` for AVX2.
`?skip_tags`, `?inline_set`, `?force_last_use` default to `None`.)

The four input parameters that need contract-checking are `scheduled`,
`inline_set`, `force_last_use`, and the relationship between `scheduled`
and the emitter's emission order.

---

## Call sites

Currently two, both inside the `(match spill with | Some sp -> ...)` branch
of `emit_codelet`:

| Call site | Location | label | scheduled | inline_set | force_last_use |
|---|---|---|---|---|---|
| Site 1 | `lib/emit_c.ml:1275` | `"spill_pass1"` | `pass1_blocked` | `Some inline_set` | `Some pass1_force_last_use` |
| Site 2 | `lib/emit_c.ml:1569` | `"spill_pass2"` | `pass2_ordered` | `Some inline_set` | `Some pass2_force_last_use` |

Both sites use the same `inline_set` (constructed once at the top of the
spill recipe; not pass-specific). `force_last_use` is pass-specific.

---

## Invariants (the contract)

### I1. Each tag appears at exactly one position in `scheduled`.

**Statement.** For every tag `t` defined by some node `e` in `scheduled`,
there is exactly one position `i` such that `(List.nth scheduled i).tag = t`.

**Why it matters.** The allocator processes `scheduled` linearly:
`List.iteri` walks each position once and at each position, it calls
`Hashtbl.replace allocated e.tag color` (consuming a color from the pool)
*without* returning the previous color to the pool. If a tag appears
twice, the second occurrence allocates a fresh color while leaking the
first one — corrupting `free_pool` for all subsequent positions.

**How site 1 satisfies it.** `pass1_blocked` is constructed by filtering
`pass1_nodes` (output of `classify_passes`) and reordering by `min_slot`.
Each tag appears exactly once because `pass1_nodes` is itself the result
of `List.filter` over the topo-sorted `nodes`, and `topo_sort_reachable`
deduplicates by construction (uses a `seen : (int, t) Hashtbl.t` table
keyed on `e.tag`).

**How site 2 satisfies it.** `pass2_ordered` is constructed similarly
from `pass2_nodes`. Same dedup property.

**How M7's SU branch violated it.** The SU scheduler's output is
`(oref_opt, e) list` where the same `e` may appear with `oref_opt=None`
(intermediate) AND with `oref_opt=Some oref` (store sink). Passing
`List.map snd scheduled` to the allocator produces a list with duplicates
when both forms exist for the same tag. **R=3 AVX-512 had ~6 such duplicates;
the resulting pool corruption produced the t0/t30 zmm1 conflict.**

**Assertion shape.**
```ocaml
let seen = Hashtbl.create 256 in
List.iter (fun (e : Algsimp.t) ->
  if Hashtbl.mem seen e.tag then
    failwith (Printf.sprintf
      "Regalloc.allocate: tag %d appears multiple times in scheduled \
       (use deduplicated emission order, not raw SU output)" e.tag);
  Hashtbl.add seen e.tag ()
) scheduled
```

---

### I2. `scheduled` is in emission order.

**Statement.** Position `i` in `scheduled` is the position at which the
emitter will emit the corresponding node. The allocator's
`current_emit_position := i` setting (done by the emitter, but consumed
by the allocator's reload-site lookup via `Hashtbl.find_opt
alloc.reload_sites !current_emit_position`) must match.

**Why it matters.** The allocator computes `last_use[t]` by walking
`scheduled` and recording the position at which each tag is referenced
(as a pred). `release_dead i` then frees tags whose `last_use < i`.
If the emitter walks a *different* order than `scheduled`, the position-
based liveness analysis is meaningless: a tag might be "dead" at
position `i` in the allocator's view but still in active use in the
emitted C at position `i`.

**How sites 1 and 2 satisfy it.** The emitter walks `pass1_blocked` /
`pass2_ordered` (the same lists passed to install_alloc) via
`List.iteri (fun pos e -> current_emit_position := pos; ...)`. By
construction, allocator position and emitter position are the same
list-index.

**How M7's Bisection/SU branches might violate it.** If the emitter uses
a dedup table to skip duplicate emissions (which Bisection and SU both do
via `let defined : (int, unit) Hashtbl.t`), the emitter's *effective*
position advances differently from the allocator's. Even if the duplicates
are removed before passing to install_alloc (fixing I1), this asymmetry
between the input list and the emit walk must be resolved.

**Assertion shape.** Cannot be checked at allocator entry alone — requires
either:
- a callback the emitter calls per-position to verify, OR
- the canonical prep pass (Stage 3) guaranteeing that the same list is
  passed to install_alloc and walked by the emitter.

Stage 2 will not check I2 directly; Stage 3 will eliminate the failure
mode structurally.

---

### I3. All IR predecessors of every node in `scheduled` are reachable
through `scheduled` ∪ `inline_set`.

**Statement.** For every node `e` in `scheduled`, and every `p ∈ preds(e)`:
either `p ∈ scheduled` (as one of the entries), OR `p.tag ∈ inline_set`
with `p` transitively reachable through inline_set predecessors that
eventually all land in `scheduled`.

**Why it matters.** The allocator computes `last_use[p.tag] := i` whenever
it encounters `p` as a predecessor at position `i`. If `p` is not in
`scheduled` and not in `inline_set`, the walk never sees `p`, and
`last_use[p.tag]` defaults to `p`'s own definition position — which may
be far earlier than `p`'s actual last reference. The allocator then
frees `p`'s color prematurely.

For the cluster-spill recipe, constants are HOISTED OUT of `pass1_blocked`
and `pass2_ordered` (constructed by `let const_nodes = List.filter is_const
nodes in let pass1_nodes = List.filter (fun e -> (not (is_const e)) && ...)`).
Constants are emitted in a pre-scope (outside both pass blocks). The
allocator never sees them.

This means: **the allocator does not allocate registers for constants under
the cluster-spill recipe.** Constants are handled by the emitter
out-of-band; gcc allocates registers for the (Const → variable) lowering.

**How sites 1 and 2 satisfy it.** Both pass `pass{1,2}_nodes` which exclude
constants AND exclude any node whose preds reach outside this pass
(cross-pass values round-trip through spill arrays, so the allocator
doesn't see cross-pass predecessor relations). The transitive walk
through `inline_set` covers single-use intermediates within the pass.

**How M7 violated it.** The Topological branch passed `nodes` (the full
topo-sorted DAG including constants) to `install_alloc`. Constants
appeared in `scheduled`, got allocated colors, and were then subject to
`release_dead`. This is a different failure mode than I1 but came up in
testing: the constant `t0` in R=3 AVX-512 was allocated zmm1 at pos 1,
then "looked alive" by the allocator until pos 31 (its last forward
reference), but the duplicate-entry corruption (I1) caused its color
to be freed early. If I1 were fixed but constants were still in
`scheduled`, the allocator would be doing work it doesn't need to do —
and the constant emission path would need to coordinate with the
allocator's color choices.

**Assertion shape.**
```ocaml
let scheduled_tags = Hashtbl.create 256 in
List.iter (fun (e : Algsimp.t) ->
  Hashtbl.add scheduled_tags e.tag ()) scheduled;
let in_scope tag =
  Hashtbl.mem scheduled_tags tag ||
  (match inline_set with
   | None -> false
   | Some s -> Hashtbl.mem s tag)
in
List.iter (fun (e : Algsimp.t) ->
  List.iter (fun (p : Algsimp.t) ->
    if not (in_scope p.tag) then
      failwith (Printf.sprintf
        "Regalloc.allocate: pred t%d of t%d is not in scheduled \
         or inline_set — caller must include all referenced tags or \
         exclude them via cross-pass spilling" p.tag e.tag)
  ) (Algsimp.preds e)
) scheduled
```

This is the strongest enforceable invariant. It catches both M7-style
violations (constants in scheduled but not handled out-of-band) and
silent-cross-pass-reference bugs.

---

### I4. `inline_set` contains only nodes with valid in-scope predecessors.

**Statement.** For every `tag ∈ inline_set`, the corresponding node `n`
(found in the IR via tag lookup) satisfies: every `p ∈ preds(n)` is
either in `scheduled` or also in `inline_set`. Inlined nodes are not
in `scheduled` themselves (by construction — emitter skips standalone
declarations for them).

**Why it matters.** The allocator's `walk_pred` recurses through inlined
preds:
```ocaml
let rec walk_pred (e : Algsimp.t) (pos : int) =
  Hashtbl.replace last_use e.tag pos;
  if is_inlined e.tag then
    List.iter (fun p -> walk_pred p pos) (Algsimp.preds e)
```
If an inlined node's pred is not in scope, last_use updates are missing
for that subtree → premature register frees.

**How sites 1 and 2 satisfy it.** The cluster-spill inline_set is filtered
to exclude:
1. Spilled tags (they need to be named for the spill store).
2. Tags with consumers in a different pass (cross-pass values are not
   inlinable — they round-trip through the spill array).

Both filters preserve I4: an inlined tag's consumers are in the same pass
as the tag's definition, so the consumers see the tag's preds within the
same scheduled list. (Constants are excluded from inline_set because they
have no preds — vacuously OK.)

**How M7 might violate it.** The SU branch's `inline_set = compute_inline_set
assigns` is *not* filtered for cross-pass relationships (primes have no
passes, so the filter is vacuous), but it IS unfiltered for spill
membership. Since primes don't spill (M7a's expected behavior), this is
also vacuous. M7's actual bug was I1, not I4.

**Assertion shape.**
```ocaml
(match inline_set with
 | None -> ()
 | Some s ->
   Hashtbl.iter (fun tag () ->
     if Hashtbl.mem scheduled_tags tag then
       failwith (Printf.sprintf
         "Regalloc.allocate: tag %d is in both scheduled and \
          inline_set — inlined tags should not appear as standalone \
          schedule entries" tag)
   ) s)
```

---

### I5. `force_last_use[t]` keys are tags that appear in `scheduled`.

**Statement.** For every `(t, pos) ∈ force_last_use`, `t` corresponds to
a node in `scheduled` (i.e., `t ∈ scheduled_tags`).

**Why it matters.** The allocator applies `force_last_use` as a lower
bound on the computed `last_use`. If `force_last_use[t]` is set but `t`
is not in `scheduled`, the entry has no effect (because last_use is
keyed on tags that were walked through scheduled). This is a silent
no-op, not a correctness bug, but it indicates caller confusion about
which tags are in scope.

**How sites 1 and 2 satisfy it.**
- Site 1: `pass1_force_last_use` is keyed on tags from `pass1_assigns`.
  `pass1_assigns` is filtered to tags in classification `Pass1`, which
  are exactly the tags in `pass1_blocked`.
- Site 2: `pass2_force_last_use` is keyed on tags from `pass2_assigns`.
  Same property for pass2.

**How M7 satisfied it.** My M7 attempt set `force_last_use` from `assigns`
(all output assignments). On the prime path, all output tags are in
`scheduled` (no cross-pass filtering needed). So I5 was satisfied; this
wasn't the bug.

**Assertion shape.**
```ocaml
(match force_last_use with
 | None -> ()
 | Some tbl ->
   Hashtbl.iter (fun tag _pos ->
     if not (Hashtbl.mem scheduled_tags tag) then
       failwith (Printf.sprintf
         "Regalloc.allocate: force_last_use entry for t%d, but tag \
          is not in scheduled" tag)
   ) tbl)
```

---

### I6. `force_last_use[t] ≥` t's natural last_use position.

**Statement.** For every `(t, forced_pos) ∈ force_last_use`, `forced_pos`
should be ≥ the position of t's last DAG-reachable use. (If `forced_pos`
is *less*, the force is a no-op because the lower-bound max takes the
natural value.)

This isn't strictly required for correctness — the allocator code applies
force_last_use as a lower bound — but it's a sanity check on caller intent.

**How sites 1 and 2 satisfy it.**
- Site 1: `pass1_force_last_use` sets all pass1_assigns tags to `pass1_n`
  (= `List.length pass1_blocked`), which is one past the last position.
  Always ≥ any natural last_use. Satisfied.
- Site 2: `pass2_force_last_use` is cluster-aware. For each output tag t
  in cluster c, `force_last_use[t] = flush_pos_for_cluster[c]`. The flush
  position is the first position of the *next* cluster (or `pass2_n` for
  the last cluster). Always ≥ any natural last_use within the cluster.
  Satisfied.

**How M7 satisfied it.** Setting all force_last_use entries to
`List.length scheduled` is the maximum possible value; always satisfies I6.

**Assertion shape.** Optional. This invariant is a hint about caller
intent, not a correctness requirement. Skip for Stage 2.

---

## Position-space relationship between allocator and emitter

Beyond the I1-I6 invariants on `Regalloc.allocate`'s inputs, there's an
inter-module invariant that connects the allocator's internal state to
the emitter's actions:

### P1. The emitter walks `scheduled` in the same order the allocator did.

This is what Stage 3's canonical prep pass will guarantee structurally:
the prep pass produces a single canonical `scheduled` list, and the
emitter is required to use it for both `install_alloc` and the per-node
emission loop. The cluster-spill recipe satisfies this by happy accident
(emitter and `install_alloc` both use `pass1_blocked` / `pass2_ordered`).
M7's Bisection/SU branches violated it by passing `List.map snd scheduled`
to install_alloc but walking `scheduled` (with duplicates) in the emit
loop.

This is an architectural invariant rather than a function-entry assertion.
Not part of Stage 2; resolved by Stage 3's refactor.

---

## Summary

The cluster-spill recipe satisfies I1-I6 by construction:
- `pass{1,2}_blocked` are deduplicated, topo-sorted, pass-filtered node lists
- `inline_set` is filtered to exclude spilled and cross-pass tags
- `force_last_use` keys are exactly the in-pass output tags
- `force_last_use` values are guaranteed-late positions (pass end or cluster flush)

These properties are NOT stated anywhere except in this document. They
emerged from incremental development; no single commit established them
as the allocator's contract. M7 broke them inadvertently and produced
silent wrong output.

**Stage 2 deliverable.** Encode I1, I3, I4, I5 as runtime assertions in
`Regalloc.allocate`. Run the 9-case regression suite to verify the
cluster-spill recipe satisfies them (the assertions must all pass on
existing-good inputs before being usable as a check on new inputs).

**Stage 3 deliverable.** A canonical prep function that constructs
`(scheduled, inline_set, force_last_use)` satisfying I1-I6 from raw
`(scheduler choice, assigns)` input. The cluster-spill recipe and the
prime/n1 path both call it.

**Stage 4 deliverable.** Re-enable M7 (primes/n1) through the canonical
prep, with the assertions remaining active. Verify R=3..R=19 produce
bit-exact or LSB-only diffs.

**Stage 5 deliverable.** Update M_PROJECT.md to document the contract,
the prep pass, and the prime measurements.
