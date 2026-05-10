# 32. The R=128 Wall: of_expr Was Walking Textual, Not Unique

## Context

During the split-radix research arc (see
[31_split_radix_research_arc.md](31_split_radix_research_arc.md)) we
attempted to wire R=128 as a single codelet and discovered the IR
pipeline didn't terminate in reasonable time. Empirical scaling per
doubling of N:

```
R=16:    0.075s
R=32:    1.243s     (16.6× per 2× N)
R=64:   20.652s     (16.6× per 2× N)
R=128:  ~5+ minutes (extrapolated)
```

Roughly O(N⁴). At the time we deferred this and pursued split-radix
instead. This doc closes the loop: the wall has been removed.

## Diagnosis

Per-pass profiling (`bin/profile_pipeline.ml`) at R=64:

| Stage                | Time |
|----------------------|------|
| `dft_expand_twiddled`|   0.000s |
| `of_assignments`     |  **24.429s** |
| `dedup_sub_pairs` #1 |   0.001s |
| `factor_common_muls` |   0.000s |
| `factor_by_atom`     |   0.000s |
| `dedup_sub_pairs` #2 |   0.002s |
| `share_subsums`      |   0.000s |

**The entire scaling cost was in `of_assignments`.** Every other pass
ran in microseconds. We had assumed `share_subsums` or the FP transpose
loop was the culprit (those are the algorithmically heaviest passes
on paper); empirically neither matters.

`of_assignments` calls `of_expr` recursively to convert raw `Expr`
trees to algsimp `t` nodes. The smart constructors at the bottom
(`mk_const`, `mk_load`, etc.) hashcons their results, so the *final*
DAG has only a few thousand unique nodes. But `of_expr` itself was
walking every textual occurrence of every Expr subtree, doing work
proportional to the textual count.

The `bin/sr_diag.ml` diagnostic shows the textual / unique ratio
exploding with N:

| R   | Textual nodes  | Unique post-hashcons | Redundancy ratio |
|-----|----------------|----------------------|------------------|
| 8   | 21,824         | 426                  | 51×              |
| 16  | 362,112        | 1,146                | 316×             |
| 32  | 5,893,376      | 2,874                | 2,050×           |
| 64  | 95,083,008     | 6,906                | **13,768×**      |

Redundancy grows ~6.5× per doubling of N. At R=64, of_expr was being
called 95 million times to produce a DAG with only ~7000 unique
nodes.

## Why the redundancy exists

The math layer (`Dft.dft_ct`) computes intermediate values like
`pass1_re.(n1_idx).(k2)` once and stores them in a matrix. PASS 2
then reads each element multiple times (each element appears in
`n2` separate output combinations). The reads return the SAME OCaml
allocation — the array slot holds an immutable Expr value, references
to it are physically equal.

Hash-consing at the leaf level catches some sharing (atomic Const and
Load nodes are deduplicated). But the recursive structure of of_expr
doesn't memoize: each textual occurrence of `pass1_re.(n1_idx).(k2)`
triggered a full walk down to its atoms, redundantly.

## The fix

Memoize `of_expr` on physical Expr identity. The same OCaml allocation,
seen multiple times, gets processed once.

```ocaml
module ExprPhysHash = struct
  type t = Expr.expr
  let equal = (==)              (* physical equality *)
  let hash = Hashtbl.hash       (* bounded-depth structural hash *)
end
module ExprMemo = Hashtbl.Make(ExprPhysHash)

let of_expr_memo : t ExprMemo.t = ExprMemo.create 1024

let rec of_expr ?(reassoc = true) (e : Expr.expr) : t =
  match ExprMemo.find_opt of_expr_memo e with
  | Some t -> t
  | None ->
    let result = (* ... existing match body ... *) in
    ExprMemo.add of_expr_memo e result;
    result
```

The memo is cleared in `Algsimp.reset()` alongside the hashcons table.

Physical equality is safe here: the dft.ml construction style stores
each intermediate Expr in an array slot once and references it through
that slot. References are physically equal. Worst case: a structurally-
equivalent-but-distinct allocation misses the memo and falls through
to a full re-walk — correct, just no speedup for that subtree.

## Impact

`of_assignments` time after the fix:

| R    | Before    | After    | Speedup |
|------|-----------|----------|---------|
| 16   | 0.095s    | 0.001s   | 95×     |
| 32   | 1.313s    | 0.006s   | 219×    |
| 64   | 24.429s   | 0.052s   | **470×**|
| 128  | ~5+ min   | 0.105s   | **~3000×**|

End-to-end gen_radix runs (full pipeline including emit_c / scheduling
/ regalloc):

| R    | End-to-end | Vec instr |
|------|-----------|-----------|
| 16   | 0.020s    | 227       |
| 32   | 0.073s    | 582       |
| 64   | 0.066s    | 1,412     |
| 128  | 0.223s    | 3,318     |
| 256  | 1.191s    | 7,842     |
| 512  | 5.251s    | 18,106    |
| 1024 | 25.2s     | (39,297 lines) |

End-to-end scaling is now approximately **O(N²·²)**, down from O(N⁴).
R=128 / R=256 / R=512 / R=1024 all generate cleanly and compile
cleanly with gcc-13.

## Validation

- ✅ Build clean
- ✅ Prime correctness 56/56 PASS (algsimp output functionally correct)
- ✅ Reachable node counts unchanged at R=16/32/64 (227, 582, 1412
   pre- and post-fix → algsimp produces the same DAG, just faster)
- ✅ R=8/16/32/64 codelets compile clean
- ✅ SR vs CT bit-exact correctness preserved (PASS at R=8/16/32/64)
- ✅ R=128 codelet compiles clean
- ✅ R=256, R=512 codelets compile clean

## What remaining scaling cost is

The fix removes the O(N⁴) wall in algsimp. The remaining ~O(N²)
scaling is in the post-algsimp pipeline: emit_c, scheduling,
register allocation. These weren't the bottleneck before (O(N⁴)
dominated everything) but they become visible at R≥256.

At R=1024, of_assignments took ~7s of the 25s total. The rest is in
emit_c / regalloc / scheduling, which haven't been profiled here
and are tractable optimization targets if/when we want to push to
R=2048+.

## Implications for the design space

R=128 monolithic was previously infeasible. Now it generates in
0.2s. This unlocks:

1. **Real benchmarking of monolithic R=128 vs decomposed multi-stage
   plans** (CT(8,16) vs 4×32 vs 8×16, etc.), which we couldn't run
   during the original investigation because the codelets didn't
   exist.

2. **The MKL-comparison question reopens with new resolution.** Earlier
   we concluded MKL likely doesn't use R=128 monolithic based on IPC
   arguments. With our R=128 codelet now generatable, we can measure
   directly.

3. **Recipe machinery stress-test on a larger boundary.** R=128 has
   ~3300 vector instructions; the spill recipe at this size is an
   open question (currently we fall back to the no-recipe path
   for sizes without explicit picker entries; the `Cooley_Tukey
   (8, 16)` recipe path needs validation).

4. **Diagnostic tools become tractable for larger scopes.** The
   `sr_structural_diff` and `sr_union_probe` tools work on the
   post-algsimp DAG; they'd been bounded by R≤64 because larger
   sizes wouldn't generate. They now work up to R=512 or beyond.

## Lessons

**Profile before optimizing.** The conventional wisdom about FFT
codelet generation (and our prior speculation) said `share_subsums`
or `factor_common_muls` would be the O(N³+) culprits because they're
the algorithmically heaviest passes. They weren't. The bottleneck was
in the boring step — IR construction — which had a quadratic-explosion
bug hidden by hashcons working at the leaf level.

**Memoization on physical identity is cheap insurance.** Whenever a
recursive function processes an immutable algebraic data type that
might be shared via OCaml's reference semantics, a `Hashtbl.Make`
with `(==)` equality and bounded-depth hashing is a 10-line addition
that catches whatever sharing is actually present. The cost when
sharing is absent is one hashtable lookup per call (a few percent
overhead at most). The cost when sharing is dense, as here, can be
the difference between O(N⁴) and O(N) on the relevant axis.

**The "we hit a wall" story is often a missing memo.** A pipeline
whose passes look reasonable in isolation but compounds badly with
input size is suspicious for redundant work. The standard diagnostic
— time each pass independently, count "textual" vs "unique" inputs
— pinpoints where the compounding lives.

## See also

- [31_split_radix_research_arc.md](31_split_radix_research_arc.md)
  — the arc that surfaced this scaling issue
- `bin/profile_pipeline.ml` — per-pass timing diagnostic
- `bin/sr_diag.ml` — textual vs unique node counter
- `lib/algsimp.ml` — the memo (search "OF_EXPR MEMOIZATION")
