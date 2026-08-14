# plan_executors.h re-promotion — OPEN (discovered 2026-08-14 during M10a)

STATUS: **OPEN — deliberate, announced re-promotion required.** Do NOT let it
ride along with unrelated work.

## The fact

`generator/generated/plan_executors.h` is **stale against its own input**:

| file | last touched | commit |
|---|---|---|
| `generated/spike_wisdom.txt` (the input) | 2026-08-13 | `a1e9d932` — "Add kind-5 wisdom for zr2c route selection" |
| `generated/plan_executors.h` (the output) | 2026-07-22 | `40c48156` — "new dag integration" |

Re-running the promote rule (`emit_executor_h.exe --wisdom spike_wisdom.txt`)
produces **+1172 / −52 lines** of executor-table change — the zr2c/kind-5-era
wisdom rows never made it into the emitted table.

It was re-promoted and **deliberately reverted** during M10a: the corpus
inversion's discipline is that no tracked-file byte change rides a structural
step. The staleness predates all restructure work.

## The standing hazard

The promote rule fires on **every `dune build @default`** in the generator —
the diff will keep resurfacing as a modified tracked file. Until this item is
taken, treat any `plan_executors.h` modification in `git status` as THIS item,
and keep it out of unrelated commits.

## The procedure (when taken)

1. Clean git state in `generated/`.
2. `dune build @default` (WSL, scoped per the usual discipline) — the rule
   re-emits and promotes the header.
3. **Review the diff before anything else**: expect additive executor rows
   derived from the post-07-22 wisdom lines (zr2c kind-5 among them); account
   for each of the ~52 removed lines (replaced/re-sorted rows are fine,
   disappearing executors are not).
4. **C-side gate**: rebuild the C library (both build systems see this header)
   and run the standing correctness gates — fd-gate (38/38), `mt_c2c_gate`,
   the c2r matrix gate — plus a canonical-bench smoke (`--zr2c` arm) to confirm
   the zr2c route still resolves. Correctness gates need no quiet machine;
   skip perf claims unless the machine is quiet.
5. Commit **alone**, named for what it is (e.g. "Re-promote plan_executors.h
   against zr2c-era wisdom"), never folded into other work.

Effort: ~30 min plus the C rebuild. Blocked on nothing.
