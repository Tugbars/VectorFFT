# Step 2 (emit_c mirror de-dup) — status + open questions

The file I'm actively working in is **`generator/lib/codelet_oop.ml`** (the
OOP-path codelet emitter), with shared helpers landing in
`generator/lib/emit_c.ml` and `generator/lib/algsimp.ml`. All three are
attached / in the tarball. Build is green; both regression gates pass.

## What step 2 is

Collapse the "Mirror of emit_c.ml line NNNN" hand-copies in `codelet_oop.ml`
into shared single-source helpers, so the OOP path and the in-place path
can't silently diverge (the failure class that produced the M-gate bug).
Each extraction is gated by two checks, run after every change:

1. **Byte-identical regen.** Regenerate the 36-codelet baseline; aggregate
   sha256 must stay `c3a6b57c3fd3a785bf14b76166d7c1bb992d696db318ecb0b09c19a24da0aaa7`
   and the per-file manifest (`/tmp/step2_baseline.manifest`) must diff clean.
   The refactor is behavior-preserving, so the emitted C must not change at all.
2. **Numeric gate.** `benchmarks/run_t1_twiddle_gate.sh` must stay 24/24 PASS.

## Done so far (2 of 8), both gate-clean

| mirror | description | resolution |
|---|---|---|
| inline_set filter | spill-path cross-pass inline filter (~40 lines, duplicated in both files) | extracted to `Emit_c.filter_inline_set_cross_pass`; both callers use it |
| topo-sort | reachable-node topo sort (~15 lines) | homed `Algsimp.topo_sort_reachable` (preds-based) at the base layer; codelet_oop calls it |

Both caught real value: the inline filter was the literal "we replicate that
filter here" drift; the topo-sort comment **lied** ("Mirrors
emit_c.topo_sort_reachable") — codelet_oop's copy used `Algsimp.preds`
(NK_Plus-tolerant) while emit_c's version is NK_Plus-fatal (the repo-wide
"fail loud during migration" guard). Naively calling emit_c's version would
have injected a migration guard into the OOP path. Homing a preds-based version
in Algsimp avoided that and left emit_c's 11 guarded call sites untouched.

A placement correction worth keeping: the original plan said "move inline_set
to Pipeline." But `Pipeline -> Emit_c` already exists (pipeline.ml uses
`Emit_c.spill_info`/`make_spill_info`) and the filter's primitives
(`compute_inline_set`/`is_spilled`/`classify_passes`) are all emit_c-resident.
So emit_c is the layering-correct home, not Pipeline. The base-layer traversal
(topo-sort) correctly went one level lower, into Algsimp.

## Remaining mirrors + my honest cost/benefit read

Current line refs in `codelet_oop.ml` (post-shift):

| line | mirror | size | drift risk | extraction risk | my call |
|---|---|---|---|---|---|
| :805 / :865 | min_slot + cluster-split SU (PASS 1) | ~80 lines | **low** (mechanical plumbing, no policy/magic numbers) | **medium** | marginal — see Q1 |
| :1177 | PASS-2 cluster-boundary store flush (emit_c 2114-2140) | ? unread | ? | ? | read before deciding |
| :926 | fused-tag handling | ? unread | ? | ? | read before deciding |
| :1332 | regalloc preconditions (emit_c 1322-1329/1349-1355/1366-1372) | ? unread | **likely high** (this is the M-gate area) | ? | **probably highest-value remaining** |
| :724 | emit_body_spill PASS1/PASS2 split (wholesale) | large | ? | high | last / decompose |

Things I found that change the naive "extract all 8" plan:

- **The cluster-SU block is pure, not emission.** It computes a reordered node
  list (`pass1_blocked`) and writes nothing to the buffer; emission consumes it
  afterward. So it's extractable as a pure function like the inline filter — no
  buffer threading. BUT emit_c's version is richer than codelet_oop's: it
  branches on a `scheduler` variant, supports `bb_budget`
  (`Bb.bb_schedule_subset`) as an alternative to `su_schedule_subset`, and calls
  `record_peak_live`. codelet_oop has none of that. So the only faithful single
  source is the **common pure core** (min_slot compute + cluster-split loop)
  with a `~schedule_cluster` closure each caller supplies. That's the first
  extraction where the two callers genuinely diverge in surrounding logic.

- **uarch selection should NOT be unified.** It looks duplicated
  (codelet_oop hardcodes `vec_regs<=16 ? raptor_lake_avx2 : sapphire_rapids_avx512`;
  gen_main passes a CLI-driven uarch into emit_c), but it differs *by design*:
  the OOP path has no CLI surface so it hardcodes a default; the main path
  honors `--uarch`. Unifying would erase that intentional difference.

- **The `gh` predicate** (`vec_regs<=16 && radix>=32`) IS duplicated verbatim in
  3 spots (gen_main:245, codelet_oop:876, codelet_oop provenance ~1364), but
  it's a one-line boolean. Centralizing a one-liner adds indirection for little
  drift protection.

## Open questions for you

1. **Cluster-SU min_slot extraction — do it or skip?** It's the largest
   remaining block but the *lowest* drift risk (pure mechanical plumbing, no
   policy), and the *highest* extraction risk so far (closure param; emit_c's
   bb_budget/peak_live/scheduler-mode branches must be preserved exactly). The
   ratio is worse than the two done. Worth the careful gated sub-step, or leave
   it (it has never drifted in a way that changed output) and spend the risk
   budget on the M-gate-area mirrors instead?

2. **Prioritize by drift risk, not file order?** I haven't read :1332 (regalloc
   preconditions) yet, but that's literally the M-gate drift area — the place
   duplication already cost us once. My instinct is to read :1332 next and treat
   it as the highest-value remaining target, ahead of the mechanical cluster
   code. Agree?

3. **The `gh` one-liner and uarch selection** — my read is leave both
   (gh: too small to be worth centralizing; uarch: differs by design). Confirm,
   or do you want gh centralized anyway for consistency?

4. **Stale-comment cleanup** — still pending from the amended plan: the line-495
   "NO scheduling, NO register allocation" header and the emit_body_spill "no SU
   scheduler" preamble that misled the earlier read, plus the now-stale ":505 t1
   fixup" comment (item 2 proved the addressing correct). Fold these deletions
   into whichever extraction touches those regions, or do one dedicated
   comment-cleanup pass at the end?

My lean: (Q2) read :1332 next, (Q1) defer the cluster-SU extraction unless
:1332 turns out low-value, (Q3) leave both, (Q4) one dedicated cleanup pass at
the end so the deletions are easy to review together.
