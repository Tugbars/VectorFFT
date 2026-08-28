# The migration: 29 steps from a 12,043-line front door to a commit surface

**Scope.** The ordered steps that move racing, planning, execution and threading logic out
of `src/core/vfft.c` into module-owned headers. Companion to
[refactor_safety_harness.md](refactor_safety_harness.md), which defines the checks; this
document defines the steps and, for each one, **the only artifacts it is allowed to
change**.

**Why the expected-diff set matters more than the step list.** The harness ladder tells you
what to run. It does not tell you whether a red result is a bug or the point of the step.
Every step below carries its own expected-diff set, and anything outside that set is a
revert — no triage in place.

**Status.** Validated against the tree at `14d366e9` (clean). Every symbol named here was
confirmed present. **10,237 lines move across 29 steps.**

---

## 1. What the validation changed

The first draft of this plan had 14 steps. Re-validating each against the current tree
broke it in three places.

**Instrumentation was one step and had to become four.** "Capture the baseline before any
edit" is self-contradictory: three of the ten baseline artifacts need library code that
does not exist. `grep -r 'VFFT_FINGERPRINT\|vfft__fingerprint'` over the whole repo returns
exactly one hit — the harness document itself. An edit is not step 1.

**`INSTRUMENT` was not a legal step class.** The harness admits `MOVE`, `MERGE` and
`DELETE` only, and says a step with no class produces an unattributable diff. Two classes
are added here: `INSTRUMENT` (expected-diff set over `src/` is empty by definition) and
`RETYPE` (a MOVE that changes a signature, so `obj_equiv` gives no signal on the touched
functions but full signal on every other symbol).

**Baseline capture from the working tree was invalid.** Every prebuilt gate binary is
**stale**: `wisdom2_real_gate.exe` 2026-08-27 21:45, `zr2c_fd_gate.exe` 2026-08-23,
against a `vfft.c` whose mtime is 2026-08-28 17:53. A baseline captured by running those
tests old code. **All 32 gates must be rebuilt at the baseline SHA before their results
mean anything.**

### Two defects in the harness this exposed

**Ladder rung 2 is theatre, and dangerously so.** `build.py:423-428` compiles with no
`-Wall`, no `-Wextra`, and with `-Wno-implicit-function-declaration`,
`-Wno-unused-function` and `-fpermissive` explicitly on. Demonstrated with the project's
exact flags: a call to a function whose declaration did not reach the caller — **the single
most likely refactor bug** — compiles **silently** and returns `1.0` where the correct
answer is `2.0`. Under `-Wall` the same code is a hard **error**. Modern gcc makes implicit
declarations an error by default; the build is actively downgrading it to nothing.
`vfft.c` is clean today, so restoring the diagnostic is free. **Until step 2 lands, rung 2
cannot fail.**

**The race census cannot be keyed as specified.** §2.4 demands keying "by function name
(never file/line)" *and* "per-site identity". Those conflict: **10 of the 24 race sites are
anonymous inline blocks inside `_vfft_create_inner`**, so all ten key to one name and
collapse the independence the census exists to check. Repair: key each site by a content
hash of its normalized skeleton (loop bound, comparison operator, aggregation callee,
alternation form) and carry the enclosing function as an *attribute*.

The enumerator also undercounts. "Functions containing a timing loop" misses
**`_calibrate_pad`** — 125 lines, six calls to `_pad_burst`, and **zero clock calls of its
own** (verified).
Three clock spellings are in use (`clock_gettime`, `vfft_proto_now_ns`, `_il_ab_now`), and
the loop variable is not always `r` (`_zt_mt_race` uses `p < 3`, the pad site uses
`rr < 3`).

### Dropped as unrepairable

**Race protocol unification.** Collapsing the 24 sites onto one round count, one reps
formula and one median is the change the drift inventory most invites — and **no rung
covers it**. `obj_equiv` is measured blind to the `.rdata` constants involved; the census
would show the change but cannot say whether the new protocol picks the same winner; and
re-racing to find out is forbidden by policy. It stays a documented fork until there is a
way to verify it. Extracting the *shared floor* (timer, median, fill) is still in — that is
steps 5 and 12–13, and it is a pure move.

---

## 2. The gating fact

**67 of the 137 function definitions in `vfft.c` take one of the three private types —
8,520 of 12,043 lines, 71% of the file** (measured; counting both the `struct
vfft_plan_s *` spelling and the `vfft_plan` / `vfft_batch` / `vfft_wisdom` typedefs).
The two giants dominate it: `_vfft_create_inner` at 4,006 lines and `vfft_execute` at 656.

So **step 15 gates roughly three-quarters of the work.** Nothing in steps 16–28 can begin until the three
structs move to a top-of-DAG internal header. The case for it is already in the tree, not
hypothetical: four bench TUs textually `#include "vfft.c"` — compiling all 12,043 lines —
precisely because they need those structs.

Everything before step 15 is deliberately a **rehearsal on struct-free code**.

---

## 3. The steps

`INSTRUMENT` = no `src/` diff. `MOVE` = byte-preserving relocation, full `obj_equiv`
signal. `MERGE`/`RETYPE` = no `obj_equiv` signal on touched functions; covered by golden
bits, census and gates instead.

### Phase A — instrumentation (steps 1–4)

| # | class | action | expected diff |
|---|---|---|---|
| 1 | INSTRUMENT | Capture the ten capturable baseline artifacts | new files under `build_tuned/baseline/` + two git-plumbing lines. **Zero `src/` diff** |
| 2 | INSTRUMENT | Add a warnings-enabled second build key | `build.py` only; identity key provably untouched via `obj_equiv` |
| 3 | INSTRUMENT | Write `race_census.py`, `run_gates.py`, `harness_golden.c` | three new files under `build_tuned/` |
| 4 | MERGE | Fingerprint emitter, replay-purity counter, trig MT counter, snapshot accessor (~250 lines **added**, not moved) | five declared diffs, enumerated below |

**Step 1 preflight, mandatory:** `git check-ignore -v build_tuned/baseline/golden_bits.txt`
must return empty. It returns `.gitignore:85:*.txt` today — the negation lands first, or
the harness reports green on an ignored baseline.

**Step 4's five declared diffs.** (i) fingerprint **off**: `obj_equiv` EQUIVALENT and the
symbol census byte-identical — that pair is what keeps step 1's baseline valid. (ii)
fingerprint **on**: `vfft__fingerprint` appears in `nm`, expected and named. (iii) exactly
**one** new mutable file-scope object (the trig counter) — declared by hand, or §2.5's stop
rule fires on the harness's own instrumentation. It must be a tentative definition in
`vfft.c` with `extern` in the trig headers, **never a `static` in a header**. (iv)
`race_census.txt` gains **zero** sites. (v) golden bits, refusal matrix and accuracy
reference all byte-identical.

### Phase B — the free moves (steps 5–14, 1,809 lines, no ruling needed)

| # | target | lines | what moves |
|---|---|---|---|
| 5 | `support/race_timing.h` | 16 | **the pilot** — the two race-timing primitives |
| 6 | `transforms/fft2d/il2d_cols.h` | 581 | IL2D column kernels, chain enumerator, table builders |
| 7 | `engine/mt_execute.h` | 148 | generic K-split MT executor + trampoline |
| 8 | `transforms/natorder/natorder_mt.h` | 142 | natural-order MT reorder passes |
| 9 | `oop/oop_mt.h` | 107 | OOP slice helpers + MT dispatch |
| 10 | `oop/zturn_mt.h` | 145 | zturn cascade MT tile/phase kernels (not the racer, not the counter) |
| 11 | `transforms/real/real_route_race.h` | 147 | r2c/c2r route racers (not the deciders) |
| 12 | `planning/cascade_calibrate.h` | 213 | zsplit/zturn terminator calibrators |
| 13 | `planning/pad_calibrate.h` | 244 | pad-vs-tail calibrator + stride helper |
| 14 | `transforms/conv/il_layout.h` *(existing)* | 82 | interleaved/split boundary converters |

**Step 5 is deliberately 16 lines.** It exists to exercise every rung end to end —
`obj_equiv`, the census keying, the gate runner — on a move that reverts with one
`git checkout`. Independently validated: extracting two functions from `vfft.c` into a
header produced **1007/1007 identical symbol bodies**.

**Step 13 has a hard precondition:** `_calibrate_pad` must already appear in
`race_census.txt`. It contains no clock call, so a naive enumerator misses it. If it is
absent at precondition time, **stop and fix the enumerator** — do not move a 244-line racer
blind.

Every step in Phase B is verified struct-free, carries no mutable file-scope state, creates
no `planning/`↔`wisdom2/` edge, and pulls no `engine/stride_executor.h`.

### Phase C — the gate (step 15)

| # | class | action | lines |
|---|---|---|---|
| 15 | RETYPE | Lift `vfft_plan_s`, `vfft_wisdom_s`, `vfft_batch_s` into `src/core/vfft_internal.h` | 420 |

**Precondition: steps 1–14 all green.** This is the first step where the harness is being
trusted rather than rehearsed.

### Phase D — gated on 15 (steps 16–28, ~8,400 lines)

| # | target | lines |
|---|---|---|
| 16 | `vfft_execute.h` — execute-side c2c dispatch + trampolines | 703 |
| 17 | `transforms/fft2d/il2d_tier.h` — IL 2D real/c2c tier, MT passes, four racers | 1199 |
| 18 | `transforms/real/zr2c_build.h` — interleaved-CCE real route | 312 |
| 19 | `oop/k1_commit.h` — K=1 replay, race-and-bank, commit | 654 |
| 20 | `oop/zturn_mt.h` *(extended)* + `transforms/fft2d/plane_queue.h` | 274 |
| 21 | **state identity** — extern-ize four counters + six decision inputs; fix the live bench split | 40 |
| 22 | `transforms/fftnd/fftnd_create.h` — rank-3/4 create | 153 |
| 23 | `transforms/fft2d/fft2d_create.h` — the dims==2 tier, largest single slice | 1249 |
| 24 | `oop/c2c_ip_create.h` — c2c in-place, padded and unpadded | 1103 |
| 25 | `oop/c2c_oop_create.h` — c2c out-of-place | 760 |
| 26 | `transforms/real/real_create.h` — r2c/c2r create + odd-real bridge | 320 |
| 27 | `transforms/trig/trig_create.h` — trig create + builders | 216 |
| 28 | `vfft_execute.h` *(extended)* + `vfft_batch.h` — execute dispatch, signature enforcement, destroy, owned-batch allocator | 1213 |
| 29 | DOCS — module README inventories | 0 |

**Step 21 fixes a live bug, not just a hazard.** `bench_1d_vs_mkl.c` includes
`r2c_dispatch.h`/`c2r_dispatch.h` while `build.py` compiles `vfft.c` separately, so its
calls to `vfft_r2c_dispatch_set_decouple_min_k` and `vfft_c2r_path_load` write a *different
copy* than `vfft_create` reads. `VFFT_C2R_PACK_ALL` and `VFFT_C2R_STRIDE_ALL` are inert
today — both probe arms measure the same route. Step 15 is its precondition because it
gives the four `#include "vfft.c"` TUs an alternative spelling.

**Step 27 is the least-protected step in the plan.** The trig family has **zero banked
wisdom** (the store holds 539 cells: 461 `t=c2c`, 59 `t=r2c`, 25 `t=c2r`, no trig), so
replay covers none of it. Its protection is entirely step 3's naive-reference accuracy
check and step 4's trig MT counter. Both are non-negotiable preconditions.

**The create tiers come last (22–27)** because each is a slice out of one 4,009-line
function, and every slice shifts the ones after it.

---

## 4. Stop rules

Uniform: **any diff outside the step's declared expected set → revert, do not triage in
place.**

Step-specific rules worth stating separately:

- **Any golden-bit change → revert.** Every step class, no exception. A refactor that
  changes an output bit is not a refactor.
- **`obj_equiv` non-EQUIVALENT on a MOVE → revert.** Not investigate — revert.
- **Any new mutable file-scope object in a module header → revert** (step 4's trig counter
  is the one declared exception).
- **Replay purity count non-zero → revert.** A cell that races under replay has a clock
  inside the baseline.
- **An engagement counter delta that is not strictly positive → revert.** A bitwise MT==ST
  pass is vacuous if threading never engaged.
- **Any `stride_execute_*` symbol in a new object's census → revert** — the forbidden
  header got pulled.
- **Any new `planning/`↔`wisdom2/` include edge → revert.**

---

## 5. What to do first

Steps 1–5 are safe today and worth doing regardless of whether the rest proceeds:

1. **Baseline capture** — zero `src/` edits, plus two mechanical git-plumbing fixes to a
   confirmed defect.
2. **The warnings key** — without it the cheapest rung in the ladder cannot fail, and the
   most likely refactor bug compiles silently to a wrong answer.
3. **The three tools** — `race_census.py` is the only thing that covers what `obj_equiv` is
   measured blind to.
4. **The instrumentation** — sanctioned by the harness doc itself.
5. **The 16-line pilot** — proves the whole apparatus on a move that reverts instantly.

That is 1,809 lines of free moves behind it (steps 6–14) before anything needs an owner
decision.
