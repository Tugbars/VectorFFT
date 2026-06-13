# Stage 2 Complete: Contract Assertions in Regalloc.allocate

## What was added

Two runtime assertions at the top of `Regalloc.allocate`, encoding the
two invariants that survived empirical testing:

- **I1**: Each tag appears at most once in `scheduled`. The M-project
  motivating bug.
- **I5**: `force_last_use` keys must be tags that appear in `scheduled`.
  Sanity check.

71 lines added to `lib/regalloc.ml`, all in a single `validate_input ()`
block called at function entry. Failures raise `Failure` with descriptive
messages identifying the offending tag and explaining the violation.

## What was learned

Stage 1 documented six potential invariants. Empirical testing against
the 9-case regression suite reduced this to two. The other four:

- **I3** (all preds in scope): legitimately violated by cross-pass
  references. Pass 2 nodes reference pass 1 spilled values via the
  spill array; the Algsimp DAG retains the pred relation but the pred
  is in a different pass's `scheduled` list. The allocator handles this
  correctly via silent ignore. The check would generate false positives
  without spill_info context the allocator doesn't have.
- **I4** (inline_set disjoint from scheduled): legitimately violated by
  the cluster-spill recipe, which places inlinable tags in both lists.
  The allocator's `is_inlined` check correctly skips allocation while
  using positions for release_dead bookkeeping. Stage 1 misread the code.
- **I2** (emission order matches scheduled order) and **P1** (emitter
  walks same list passed to allocate): inter-module invariants resolved
  structurally in Stage 3.
- **I6** (force_last_use values are guaranteed-late): hint-only, not
  required for correctness.

The Stage 1 doc has been left as-is (with its incorrect I3 and I4
analyses) so the empirical-finding paper trail is preserved. The
Stage 2 deliverable corrects the analysis in the validate_input block's
docstring, with cross-references back to the Stage 1 invariant numbers.

## Verification

Three checks:

**1. Existing 9-case regression suite passes with assertions active.**

| Codelet | Result | Tag counts |
|---|---|---|
| R=64 AVX-512 t1_dit | PASS | 796 + 518 |
| R=64 AVX-512 t1_log3 | PASS | 796 + 518 |
| R=128 AVX-512 t1_dit | PASS | 1916 + 1046 |
| R=128 AVX-512 t1_log3 | PASS | 1916 + 1046 |
| R=256 AVX-512 t1_dit | PASS | 3836 + 2758 |
| R=256 AVX-512 t1_log3 | PASS | 3836 + 2758 |
| R=32 AVX2 t1_dit | PASS | 396 + 174 |
| R=64 AVX2 t1_dit | PASS | 796 + 518 |
| R=16 AVX2 t1_dit | PASS | 156 + 86 |

**2. M7's silent-wrong-output bugs are now loud failures.**

Reapplied the M7 attempt and ran the prime cases that previously
produced silent wrong output:

| Case | Pre-Stage-2 behavior | Post-Stage-2 behavior |
|---|---|---|
| R=3 AVX-512 | wrong output, no error | `Failure(I1): tag 39 appears multiple times` |
| R=5 AVX2 | compile fail (regalloc_spill undeclared) | `Failure(I1): tag 96 appears multiple times` |
| R=19 AVX2 | "no eviction candidate" hard crash | `Failure(I1): tag 1137 appears multiple times` |

All three M7 failure modes now stop at the assertion with a clear error
message pointing at the actual root cause (duplicate entries from raw SU
scheduler output passed to `install_alloc`).

**3. R=3 AVX2 (the one M7 case that worked) still works.**

R=3 AVX2 used the Topological branch which doesn't have the duplicate-entry
issue. With Stage 2 assertions + M7 emit code, R=3 AVX2 still produces
28 register pinnings and passes the assertion check.

## Files

- `/mnt/user-data/outputs/regalloc_stage2.ml` — full updated regalloc.ml
- `/mnt/user-data/outputs/regalloc_stage2.diff` — 71-line diff vs pre-M7

The tree state after Stage 2:
- `lib/regalloc.ml`: contains Stage 2 assertions (always active, no gate)
- `lib/emit_c.ml`: reverted to pre-M7 baseline (no prime support)
- M7 attempt preserved at `/tmp/emit_c_m7_attempt.ml`

The 9 cluster-spill cases continue producing the same output as before
Stage 2 — the assertions add no runtime overhead beyond a single Hashtbl
walk at allocator entry, and they pass on every existing call site.

## Architectural significance

This is the **first time** the allocator's input contract is checked at
runtime. Before Stage 2:

- The contract was implicit, documented nowhere
- Misuse produced silent wrong output (the R=3 AVX-512 bug)
- 9/9 correctness on composites was "lucky" — the only caller happened
  to construct valid inputs

After Stage 2:

- The contract is explicit (REGALLOC_CONTRACT.md) and partially checked
- Misuse fails loudly with diagnostic messages
- Any new caller violating I1 or I5 is caught immediately
- The "lucky" property becomes "correctly-by-assertion" for the existing
  caller

The remaining failure modes (cross-pass pred references, emission-order
divergence) can't be checked from inside the allocator without more
caller context. Stage 3 (canonical prep pass) will resolve them
structurally by ensuring the caller passes consistent inputs by
construction.

## Next: Stage 3

Factor out the prep work (dedupe, inline_set construction, force_last_use
population) from the cluster-spill recipe into a single function that
any caller — cluster-spill OR prime/n1 — can use. The function's output
satisfies I1 + I5 by construction, and the cross-pass/emission-order
issues that I3/I2 can't detect at the allocator level are resolved by
making the prep function the single source of truth.

After Stage 3, the prime/n1 path (Stage 4) can re-enable without risking
the M7 failures we hit.
