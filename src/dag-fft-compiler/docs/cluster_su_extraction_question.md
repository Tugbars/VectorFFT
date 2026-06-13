# Problem: should we extract the "cheap half" of the cluster-SU mirror, and how?

## Context (one paragraph)

VectorFFT's OCaml codelet generator has two emitters that share logic by
hand-copying: `lib/emit_c.ml` (in-place path) and `lib/codelet_oop.ml` (OOP
path). Step 2 of an ongoing cleanup is collapsing ~8 "Mirror of emit_c.ml line
NNNN" hand-copies in codelet_oop into shared single-source helpers, each
extraction gated by (1) byte-for-byte regeneration of a locked 48-codelet
baseline (provenance-normalized; includes `--fuse 2` variants so the fused path
isn't dead code) and (2) a numeric correctness gate. Three mirrors are already
done (inline_set filter, topo-sort, fused-tag predicate). One (`:1177` store
flush) was deliberately NOT extracted because the two sides diverge by design
(emit_c has regalloc + store-on-compute; OOP doesn't), so unifying would fight an
intentional capability gap for ~10 lines of yield. The current candidate is the
"cheap half" of the cluster-SU mirror.

## What cluster-SU does

For blocked CT(N1,N2) codelets (radix >= 25), PASS 1 computes N1 independent
sub-FFTs of size N2. Within each "cluster" (one sub-FFT), the nodes are
reordered by a software-pipelining scheduler (`Schedule.su_schedule_subset`) for
better ILP; across clusters, order is preserved (clusters are independent, so
this is safe). The same idea applies in PASS 2 with a different cluster
definition. This is the perf-bearing scheduling path for the exact codelets the
future "blocked executor" will run in its hot loop.

## The PASS-1 sub-blocks I'm proposing to extract

Having read both files in full, the PASS-1 cluster-SU code has three pieces:

**Sub-block A — min_slot reverse-topo computation.** Builds a `succs` map over
PASS-1 nodes, then walks nodes high-tag-first assigning each node a `min_slot`
(its own spill slot if it's a spill target, else the min of its successors'
min_slots), then sorts by `(min_slot, tag)` to get `pass1_blocked_topo`.

- codelet_oop and emit_c are **logically equivalent but NOT verbatim**:
  - reverse order: codelet_oop does `List.sort (compare b.tag a.tag)`; emit_c
    does `List.rev pass1_nodes`. These are equal ONLY IF `pass1_nodes` is
    ascending-tag-ordered. I verified it is in both (both come from
    `topo_sort_reachable`, which sorts by tag), so they produce the same order.
    But the equivalence rests on that invariant.
  - slot lookup: codelet_oop uses a local `lookup_slot` (re_slot then im_slot);
    emit_c inlines `match lookup_re_slot, lookup_im_slot with Some s,_|_,Some s`.
    Same logic.

**Sub-block B — contiguous-run splitter.** A `go` recursion that walks the
sorted `pass1_blocked_topo` splitting it into maximal same-cluster runs, then
`List.concat_map` over the runs: for each run, filter `cluster_sinks` (nodes
that are spill targets), and if non-empty call `su_schedule_subset uarch ~gh
~subset ~sinks`, else keep the run as-is.

- This sub-block IS **verbatim-identical** between the two files, modulo variable
  names (`cur_cluster`/`current_cluster`, `k`/`c`) and type annotations.
- ONE divergence: emit_c wraps the per-cluster scheduler call in a `bb_budget`
  match — `None -> su_schedule_subset` / `Some t -> Bb.bb_schedule_subset` —
  whereas codelet_oop only ever calls `su_schedule_subset`. This is the natural
  `~schedule_cluster` closure parameter (each caller passes its own).

## Proposed extraction shape (PASS 1 only)

Two shared functions (home: `emit_c` or a new `emit_common`, since
`Schedule`/`Algsimp` are lower layers both already depend on; `Pipeline ->
Emit_c` already exists so Pipeline is not the home):

```
compute_min_slot_pass1 : spill_info -> pass1_nodes:t list -> (int,int) Hashtbl.t
  (* sub-block A; use an EXPLICIT descending sort, not List.rev, so it does
     not silently depend on the caller pre-sorting by tag *)

cluster_split_schedule :
  spill_info -> pass1_blocked_topo:t list -> min_slot:(int,int) Hashtbl.t ->
  schedule_cluster:(subset:t list -> sinks:t list -> t list) -> t list
  (* sub-block B; caller supplies schedule_cluster.
     codelet_oop passes:  fun ~subset ~sinks -> su_schedule_subset uarch ~gh ~subset ~sinks
     emit_c passes:       fun ~subset ~sinks -> match bb_budget with
                            | None -> su_schedule_subset uarch ~gh ~subset ~sinks
                            | Some t -> Bb.bb_schedule_subset uarch ~time_budget_sec:t ~subset ~sinks *)
```

The `uarch`/`gh` selection stays per-caller (it differs by design: emit_c gets
uarch from a CLI flag via its caller; codelet_oop hardcodes
`vec_regs<=16 ? raptor_lake_avx2 : sapphire_rapids_avx512` and
`gh = vec_regs<=16 && radix>=32` because the OOP path has no CLI surface — same
"do not unify, differs by design" status as previously decided for uarch).

## The actual question(s) for discussion

**Q1 — Is sub-block A worth extracting given it's only logically-equivalent?**
Sub-block B is verbatim (clean win). Sub-block A is equivalent-under-an-invariant
(`pass1_nodes` is tag-sorted). Extracting A normalizes two textual forms into
one. The byte-diff gate will catch any divergence, but only over the inputs the
48-codelet baseline exercises. Since the tag-ordering invariant is structural
(both feed from `topo_sort_reachable`), it holds for all codelets, not just the
baseline. Is that enough to extract A, or is the right call to extract ONLY the
verbatim sub-block B (the splitter) and leave A's ~25 lines per-caller? Extracting
only B is lower-risk but leaves the larger half duplicated.

**Q2 — PASS 2 is a DIFFERENT shape; does that kill the yield?** I assumed PASS 1
and PASS 2 might share the extracted helpers. They do NOT:
- PASS 1 splits a pre-sorted list into contiguous runs (the `go` fold), sinks =
  spill-slot membership.
- PASS 2 builds `cluster_of_pass2_node` via a fixpoint, buckets into
  `Array.make ct_n2 []`, loops `for k2 = 0 to ct_n2-1`, sinks = `assign_tags`
  membership (assigns_post tags, not spill slots).
So PASS 2 uses array-bucketing + a different sink predicate; the PASS-1 splitter
helper does not serve it. If I extract PASS-1-only, I de-dup the PASS-1 mirror
(codelet_oop:811-913 vs emit_c:1826-1916) but PASS 2 stays its own thing. Is a
PASS-1-only extraction still worth it, or does the fact that the pattern doesn't
generalize to PASS 2 suggest this whole sub-area is "structurally similar but not
actually one function," and we should de-rot the comments (like :1177) rather
than extract?

**Q3 — Yield vs risk, bluntly.** The realistic yield of the PASS-1 extraction:
sub-block B (~25 lines, verbatim) + maybe sub-block A (~25 lines, equivalent) =
up to ~50 lines de-duplicated, against the introduction of a closure-parameterized
shared function and a `~schedule_cluster` indirection. Compared to the three done
(inline_set, topo-sort, fused-tag), this is more code but lower drift-risk (it's
mechanical scheduling plumbing with no policy/magic-numbers; it has never been
observed to drift in a way that changed output). The prior analysis's guidance was
"defer the full extraction, but the cheap half (min_slot + splitter, no closures)
is a smaller cut that turns medium risk into low." But on reading, the splitter
DOES need one closure (the bb_budget divergence), and sub-block A needs the
invariant argument. So it's not quite "no closures." Does the cheap half still
clear the bar, or is this another "leave it, fix the comment" like :1177 and
uarch?

## My current lean

Extract sub-block B only (verbatim, clean, one `~schedule_cluster` closure),
leave sub-block A per-caller (the invariant-dependence makes normalizing it a
latent footgun for marginal gain), scope strictly to PASS 1, and de-rot the
PASS-1 + PASS-2 "Mirror of emit_c.ml lines NNNN" comments to function-name
references in the end-pass. But I'm not confident the B-only cut is worth the
closure indirection for ~25 lines, and I could be talked into "leave both, fix
comments" if the reviewer thinks the scheduling plumbing is low-drift enough that
the de-dup doesn't pay. Want a second opinion on Q1/Q2/Q3 before cutting.

## Relevant exact locations (current, post-3-extractions)

- codelet_oop PASS-1: `lookup_slot` @806, min_slot walk ~811-849, splitter
  ~887-913, per-cluster su call @911.
- codelet_oop PASS-2: `cluster_of_pass2_node` fixpoint @1078-1138, array-bucket
  scheduling @1139-1170.
- emit_c PASS-1: pass1_nodes @1725, min_slot @1838-1862, splitter @1880-1905,
  per-cluster su/bb call @1909-1913.
- `Schedule.su_schedule_subset` @schedule.ml:692; `Bb.bb_schedule_subset` exists.
- Gate scripts: `benchmarks/snapshot_step2_baseline.sh`,
  `benchmarks/diff_step2_baseline.sh` (provenance-normalized), locked normalized
  aggregate `5e7f772c0268bdb89631139a709efc191d393d4def8c0d7c9c9969b3d70d6e87`.
