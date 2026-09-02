# The refactor safety harness: proving the rearchitecture changed nothing

**Scope.** The checks, artifacts and run protocol that gate the `src/core/vfft.c`
rearchitecture — moving ~10,000 lines of racing, planning, execution and threading logic
into module-owned headers while the front door shrinks to config admission and commit.

**Audience.** Whoever is executing a migration step, and whoever has to decide whether a
red result means "revert" or "investigate".

**The governing constraint.** This is a header-only library measured on a thermally noisy
host. Cross-session nanoseconds are not comparable, and project policy forbids re-racing
during development. **Every gate in this harness is therefore clock-free.** Exactly one
check uses a clock, it is milestone-only, and its statistic is chosen so that noise cannot
produce a false failure (§6).

**Reading rule.** Every check below **failed its first adversarial review**. What is
written here is the repaired form. Where a check's first draft is instructive — because it
looks sound and is not — the false pass is recorded with it. Do not restore a first draft.

---

## 1. The failure model

A refactor does not usually produce wrong answers. The gates already catch that: 32 gates
exist and 21 assert bit-identity. The failures that matter here are the ones a green gate
suite cannot see:

| failure | why the gates miss it |
|---|---|
| **Chose differently** — a different but still-correct plan is built | output is correct, just produced by a slower route |
| **Stopped being reached** — an arm, racer or thread path is no longer entered | the fallback is correct; only speed and coverage change |
| **Protocol drift** — a race's reps, rounds, median rank or hysteresis moved | the next calibration banks a different winner, permanently |
| **State split** — a `static` in a header gets one copy per translation unit | setter and reader silently address different objects |

The last one is not hypothetical. It is **already present** in the canonical bench:
`bench_1d_vs_mkl.c` includes `r2c_dispatch.h` and `c2r_dispatch.h` (:70-71) while
`build.py:625` compiles `vfft.c` as a second translation unit. Both headers hold mutable
`static` state with public mutators — `_vfft_r2c_decouple_min_k` (r2c_dispatch.h:97, with
a setter) and `_vfft_c2r_paths` (c2r_dispatch.h:311, with a loader). The bench writes its
own copy at :4504/:4506/:4530 and then builds through `vfft_create`, which reads
`vfft.c`'s untouched copy. Consequence: **`VFFT_C2R_PACK_ALL` and `VFFT_C2R_STRIDE_ALL`
are inert** — both probe arms measure the same route — and the calibrated c2r path table
never reaches the library.

---

## 2. The check ladder

Ordered cheapest-first. A step runs every applicable rung; the first red stops the step.

### 2.1 Step class — declared before editing

Every step is declared **MOVE** (byte-preserving relocation), **MERGE**
(behaviour-preserving rewrite) or **DELETE**, in the commit message, before the edit. This
is not a check; it is what makes the checks attributable. A MERGE gets no `obj_equiv`
signal by design, so a step with no declared class produces an unattributable diff and the
harness degrades to "we ran a tool".

### 2.2 Build and link at the baseline flags key

Zero new warnings. **A duplicate-symbol link error is a PASS of §2.6, not a failure** — it
is the linker catching a split-state hazard for free.

### 2.3 `obj_equiv.py` — code identity (MOVE steps only)

`python build_tuned/obj_equiv.py before.o after.o`

Proves every emitted symbol body is unchanged, modulo three normalizations, each derived
from an observed false positive:

1. **addresses** — offsets shift when code moves;
2. **objdump's comments** — it annotates a RIP-relative load with the *nearest* symbol,
   which changes when neighbours move (observed on `_vfft_tname`: identical instruction,
   different comment);
3. **alignment NOPs** — the assembler picks `nopl` vs `xchg %ax,%ax` by landing position
   (observed on `_dht_worker_post`).

Validated on a real extraction: **1007/1007 symbol bodies identical**. Naive comparison
does not work — the same extraction produced an `.o` of identical size with 9,303 differing
bytes (`.text` alone: 8,484).

> **Measured blind spot — do not forget it.** This tool is blind to DATA. Floating-point
> constants live in `.rdata`; only the constant's *address* appears in the disassembly. A
> `0.97 → 0.96` hysteresis change — exactly the silent-verdict-drift bug this refactor most
> fears — was reported **EQUIVALENT**. Comparing `.rdata` does not rescue it: as bytes it
> differs on any pure move; as a sorted multiset a pure move perturbs **92 of 4072** words
> while the real bug perturbs **4**. The signal sits under the noise. §2.4 is what covers
> this, and it is not optional.

Must run with `-DVFFT_FINGERPRINT` **off** — adding `vfft__fingerprint` correctly reports
an APPEARED symbol.

### 2.4 Race protocol census — a static source census, not a runtime trace

A tool alongside `obj_equiv.py` that parses each racer body **keyed by function name**
(never file/line, so it survives the move into headers) and extracts mechanically:

- every floating-point literal in a verdict comparison,
- the comparison operator (`<` vs `<=` — **which arm holds ties**),
- the aggregation callee (`_pad_med` / `_il_ab_med9` / an inline min-reduction),
- the burst-loop bound and the reps expression *as source text, including its clamps*,
- the arm-alternation form.

Diff that census as text.

**Why static and not a runtime trace.** A runtime trace only records branches that
execute. `_calibrate_zsplit_t2q` reads `if (inc == 0) win = (n1 < n0*0.97); else win =
(n0 < n1*0.97);` where `inc` is the create default — and on the only path that races, nothing
sets it to 1. The `else` arm never runs, so a trace can never see its constant. A source
census covers branches that never execute, needs no wisdom state, no thread count and no
clock, and is exactly orthogonal to §2.3's `.rdata` blindness.

Two requirements:

- **An explicit racer inventory with a count.** The enumerator is "functions containing a
  timing loop", *not* "functions containing 0.97" — otherwise `_pq_mt_race`'s bare
  `tq < tl` and `_zt_mt_race`'s bare `mt < st` are invisible to the census meant to watch
  them.
- **Per-site identity.** Do **not** hoist the constants to one shared `VFFT_RACE_HYST`.
  That would collapse independently settable sites into one and destroy the very
  independence being checked.

This census is also the current inventory of the drift it exists to stop: **12 racers, 3
different round counts (six use 5, four use 9, one uses 3), 4 reps formulas, 3 median
implementations.** `_pq_mt_race`, added most recently, uses min-of-3 with no hysteresis
while its neighbours use median-of-9 with 3%.

### 2.5 Census diffs — mutable state and symbol reach

Two sorted text censuses, diffed at **identical** flags keys, never compared across keys:

- **mutable file-scope objects per TU** (`nm`, filtered to `b/B/d/D` locals);
- **defined and undefined symbols** of the library object.

A census-and-diff, never a count-assertion: a count assertion goes red on an unrelated
`-flto` or ASan flag change, and under a stripped build degrades to a false failure. A diff
degrades to a vacuous empty-vs-empty pass, which is the correct failure direction.

**Stop rule.** Any *new* mutable file-scope object in a module header → revert.

### 2.6 State identity — let the linker be the gate

The property worth asserting is not "src/core contributes one `.c`" — a refactor that moves
`_il2d_real_cols_mt` and its counter together into `il2d_mt.h` passes that check green
while splitting the state. The property is: **the counter the accessor reads is the counter
the increment writes, in every linked binary.**

Make it structural. One tentative definition in one TU, `extern` in the header; a second
copy then becomes a **duplicate-symbol link error** — loud, automatic, portable, and
impossible to skip. Apply to the four engagement counters and to the real hazards, which
are the decision *inputs*, not the counters:

`_vfft_r2c_decouple_min_k` · `_vfft_c2r_paths` · `_stride_verbose` ·
`_stride_num_threads` / `_stride_workers` / `_stride_pool_size` · `_vaw_tab` / `_vaw_n` /
`_vaw_loaded`

Back it with a runtime address-identity probe inside the *existing* MT gates (logic in the
module header, driver stays thin) asserting the accessor's TU and the incrementing TU
report the same address.

**Prerequisite this exposes:** the four TUs that textually `#include "vfft.c"` must stop
doing so, or get a `-DVFFT_SINGLE_TU` spelling. That migration is the work that actually
removes the hazard.

### 2.7 Golden output-bit digests — the strongest artifact

A per-cell digest of every output plane's raw bytes, per direction, compared **bitwise**.

This is the check that closes what nothing else can: a selector→pointer mis-wiring during a
merge (the `il_kv` blocked variants `n1tb48`/`t2b48` differ at ~1e-16), a DCT2/DCT3
shared-plan arm swap, a DCT1/DST1 inner-M transposition. **A roundtrip cannot see any of
them** — a roundtrip through a swapped forward/backward pair still round-trips.

A hash is the wrong instrument for a plan trace and the right one here: the *cell* is the
triage unit, and a diff of 4096 doubles carries no information a cell name does not.

**Stop rule.** Any digest change → revert. No exception, on any step class, ever. A
refactor that changes an output bit is not a refactor.

### 2.8 Replay fingerprint — same decisions, given a frozen store

With `VFFT_WISDOM_DIR` pointed at a frozen copy of the store and `wisdom_write=0`, the
create-time fingerprint text for every corpus cell must be byte-identical.

**Emitted as stable named-token text**, one `@fp` line per plan node with a depth prefix,
modelled on the `@vw2` cell grammar — not a struct, not a hash. Named tokens mean `diff`
points at the field that moved, and adding a field appends a token instead of reflowing the
file. The emitter has no floating-point format specifier, so a timing cannot leak in.

Four properties are load-bearing:

- **One process per cell.** `_ord_n`/`_ord_pick` in the K=1 pair-ordering race is a process-lifetime
  memo keyed by N, consulted before the pair race; `_stride_num_threads`/`_stride_pool_size` and
  the QPC frequency are process-lifetime too. `--all` exists for triage only and writes to
  a filename no verdict reads.
- **Replay purity is an assertion, not an observation.** The corpus-wide count of
  `hit=race` must be exactly **0**. A cell that races under replay has the clock inside the
  baseline and will false-diff on the first thermal wobble. Back it with a create-time race
  counter (the engagement counters are the precedent) and fail loudly rather than diffing a
  coin flip.
- **The runner scrubs the environment.** `src/core` holds 99 `getenv` calls, many of them
  route selectors. Clear every `VFFT_*` from the child except what the cell spec names, and
  echo the surviving set in the header so contamination shows as a named diff line.
- **It is create-only.** It cannot see the execute-side dispatch — the trig switch, the
  layout branch into `vfft_c2r_disp_execute_z` vs `_execute`, the MT engage decision. A
  deleted `case` that falls to a correct `default` leaves the create fingerprint identical.
  §2.7 and §2.9 cover that; "the fingerprint is clean" must never be read as "the path is
  covered".

**Corpus generation.** Generate the replay corpus **from the store**, so coverage of banked
cells is 100% by construction and grows as wisdom grows. The store holds **539 cells: 461
`t=c2c`, 59 `t=r2c`, 25 `t=c2r`, and zero trig** — so the trig family has no banked verdict
to replay and is covered by §2.10 instead.

### 2.9 Reachability and engagement

- **Racer-fired census** — every racer fires at least once across the cold sweep, and the
  set that fired is unchanged. Catches a racer turned into an unconditional `return`.
- **Reach census** — the set of `src/core/**` functions executed by the sweep is unchanged.
  The only check that finds "a path stopped being reached" without knowing in advance which
  arm to ask about.
- **MT engagement** — MT output equals ST **bitwise** *and* the counter delta is **strictly
  positive**. Both, always: a bitwise pass alone is vacuous if threading never engaged.
  Note the coverage gap — the four counters cover the TC batch, IL2D columns, the zturn
  cascade and the plane queue. **Trig MT has no counter**, so a DCT/DST/DHT cannot move
  any of them; add the increment in the trig inner dispatch and expose all counters through
  one internal snapshot accessor declared only in the fingerprint header.

### 2.10 Reference accuracy and refusals

- **Naive-reference check** — for each of the 11 transform values, in both directions, the
  forward output matches an independent naive O(N²) evaluation of that transform's defining
  sum. This is the only thing that covers the trig family, which has zero banked wisdom, and
  the only thing that catches a shared-plan arm swap.
- **Refusal matrix** — `vfft_create`'s accept/refuse decision is unchanged for every cell of
  the declared config space, legal and illegal alike, recorded as **name-keyed sorted
  lines** (never a positional bitmap). Both directions matter: over-refusal is caught by the
  legal twin, under-refusal by the illegal cell.

---

### 2.11 Capturing the artifacts

`build_tuned/capture_baseline.py` produces both diffable artifacts. Use it rather
than a shell loop.

```
python build_tuned/capture_baseline.py --out <scratch>            # per-step compare
python build_tuned/capture_baseline.py --out build_tuned/baseline --repeat 5
```

It builds both binaries with `VFFT_FINGERPRINT=1`, runs one process per cell
against a fresh seeded scratch copy of the store, repeats each cell and writes LF.
Two independent captures are byte-identical; that is the property every per-step
comparison rests on, so verify it after changing the harness.

Four things it fixes, each of which had already produced a wrong answer:

**The purity assert can be compiled out.** `races_now()` returned `-1` without
`VFFT_FINGERPRINT` and the assert was skipped whenever the count was
unavailable — so a plain `build.py` invocation produced a harness that ran green
and checked nothing. `harness_golden` now `#error`s instead of degrading, and the
capture script sets the flag itself. A check that can be silently disabled is
worse than no check.

**The race counter is a positive signal, not a proof.** It fires only where
someone added it, and every site is in `vfft.c`; a racer defined in another header
is invisible to it and to the §2.4 census, which enumerates clock calls in
`vfft.c` plus their local callers. Nineteen headers under `src/core` call a clock.
One hole is closed (`vfft_natorder_race`, now counted at its call site); assume
others remain, which is why cells are captured repeatedly and disagreement is
recorded rather than sampled.

**Line endings.** Both binaries write stdout in binary mode. In text mode msvcrt
emits CRLF while `.gitattributes` pins the baseline to LF, so a byte-identical
result reported all 36 rows as changed.

**Cell counts come from `--list`.** Inferring one from output length made
`--cell` run past the end, where the range check declines and the all-cells path
runs again per index: 6 cells became 1268 rows. Out-of-range `--cell` is now an
error.

#### A nondeterministic cell is a fact to record, not a diff to chase

`c2c.split.ip.natural` has no banked nat entry, so it races and picks between
radix chains whose outputs differ in the last bits — two digests, roughly 5:3
across runs. Both are **correct**: checked against a naive long-double DFT, rel
err 3.0e-16 and 3.1e-16. Recording either one makes the artifact flap for a
legitimate reason, which trains you to shrug at a real diff. Raced cells are
therefore written `NOT_BANKED_RACED` and cells that differ across repeats
`NONDETERMINISTIC`. Those lines are still checks: a cell that starts or stops
racing changes them.

## 3. Baseline capture

Nothing in this list may run against a dirty tree.

| # | artifact | owner time |
|---|---|---|
| 0 | clean committed green tree; record the baseline SHA | ✔ |
| 1 | toolchain + flags key (compiler version, build invocation, ISA, ASan off) | |
| 2 | frozen wisdom store + content hash | |
| 3 | baseline object for `obj_equiv` (fingerprint **off**) | |
| 4 | `fp_replay.txt` — fingerprint trace, one process per cell | |
| 5 | `golden_bits.txt` — per-cell output digests | |
| 6 | `refusal_matrix.txt` | |
| 7 | `race_census.txt` — the static protocol census | |
| 8 | `mt_state.txt` — mutable-object census, counter addresses, engagement deltas | |
| 9 | `reach_functions.txt` — symbol census | |
| 10 | `accuracy_ref.txt` — max relative error vs naive reference | |
| 11 | `perf_baseline/` — the archived pre-refactor **binary** plus its cell list (§6) | ✔ |
| 12 | `gates_baseline.txt` — all 32 gates PASS/FAIL, no timings; flaky gate run 3× | ✔ |

**Git plumbing, before anything else:** `.gitignore:85` ignores `*.txt`. Either force-add
the baseline directory or add a negation for it, pin `eol=lf` in `.gitattributes` the way
`wisdom2_*.txt` already is, and **truncate rather than append**. The harness must run
`git check-ignore -v` and refuse to report green if its own baseline is ignored.

---

## 4. Per-step protocol

Run in order; first red stops the step.

1. Step class declared → 2. build clean → 3. `git diff -M -w` sanity (a MOVE must be
content-identical modulo whitespace and includes) → 4. `obj_equiv` (MOVE only) → 5. race
protocol census diff → 6. census diffs (mutable state, symbol reach) → 7. **golden bits
diff** → 8. fingerprint diff + replay purity → 9. refusal matrix diff → 10. engagement
(address identity + strict increase) → 11. the gates whose module the step touched.

**Stop rule, uniformly:** any unexplained diff → **revert, do not triage in place**. The
one exception is a diff written down in the step's declared expected-diff before the edit.

**Milestone only:** all 32 gates, a full cold re-derivation of the key/shard census, and
§6.

**Explicitly forbidden per step:** any timing, benchmark or re-race.

---

## 5. Coverage

A 57-cell matrix covers every value of every axis at least once: transform (all 11),
placement, layout, order, rank 1–4, batch geometry, threads (1 and 8), rigor, owned
buffers. It is deliberately small — the replay corpus supplies breadth (539 cells,
generated from the store); the matrix supplies **axis coverage the store cannot**, above
all the trig family and the illegal cells.

---

## 6. The performance leg

Clock-free checks prove decisions and bits are unchanged. They cannot prove speed. This leg
covers that, and it is the **only** check in the harness that uses a clock.

**Why it works despite the noisy host.** Thermal drift, preemption and cache pollution only
ever make a deterministic computation *slower* — nothing makes it faster than its best
case. The noise is **one-sided**, so the **minimum over N paced runs** is a robust estimator
of what the machine can actually do. "Did it ever get close to the recorded time" is the
right question; "was the average the same" is not.

**Protocol.**

- **Milestone only**, never per step — it costs machine time.
- A small cell list, fixed at baseline.
- **N ≥ 10 runs with pacing between**, take the **minimum**.
- Compare against the **archived pre-refactor binary re-run in the same session**,
  alternating with the new one — not against numbers in a document. Same-run A/B removes
  machine-state drift entirely instead of absorbing it in the tolerance.
- **Pass if the new minimum is within 10% of the old minimum.** Tighten toward the observed
  same-run control spread once that is known.
- `docs/performance/v1_0_results.md` is the fallback where no archived binary exists, and
  remains the **accuracy** reference regardless. Its ratio columns are not a bar.

**What it is.** A regression *detector*: it catches structural losses — a route that
stopped being taken, threading that disengaged, an inliner that lost a hot call. A 2×
regression cannot hide under a 10% band, and noise cannot fake its way under it.

**When a perf check goes red: re-run the planner before concluding anything.**

A large apparent regression has two very different causes, and the cheap one is far
more common: a hot or loaded machine. The DP planner selects by measurement, so on a
contended host it does not merely time badly, it *chooses* badly - a worse plan, then
a worse number. Distinguish them by re-planning the cell on a quiet machine, several
times, and taking the minimum.

**Re-plan at PATIENT, not MEASURE.** The two tiers differ in exactly the knob that
matters here. MEASURE uses beam 3 and *believes* a cached sub-plan cost - the FFTW
`BELIEVE_PCOST` analog - so a single timing taken during a hot moment is cached and
trusted for the rest of the search, propagating into every downstream decision.
PATIENT uses beam 8 and re-measures on every cache hit, so in the planner header's
own words "variance is re-absorbed on every encounter". Triage numbers taken at
MEASURE overstate the instability. Re-plan with `cfg.rigor = VFFT_PATIENT`.

Measured example, N=256 K=256 c2c in-place scrambled. Twelve cold replans:

    flat.t1s.t1s      5 runs   min 33,491 ns   chains 8.4.8 / 8.8.4 / 4.8.8
    seven other families        51,692 - 64,912 ns

Cleanly bimodal, and the run order is the tell - the sweep began right after a
32-gate build, and the good family was found in 1 of the first 8 runs and 4 of the
last 4. The planner was not weak; it was measuring on a hot machine. An earlier
single run under active load reported 123,384 ns, which would have looked like a 3x
regression and was nothing of the kind.

Two rules fall out:

* **Discard the first runs after any load.** Do not average them in. Minimum-of-N
  handles this on its own, which is one more reason the statistic is the minimum and
  never the mean.
* **An unstable cell is not quarantined, it is investigated.** A cell whose plan
  moves under recalibration is the one most worth re-planning, not the one to drop
  from the corpus. Exclusion would discard exactly the cells that carry information.

The same sweep incidentally showed the banked plan for that cell is not the best
available - `8.4.8` beats the banked `4.4.16` by 18%, and `4.4.16` re-measured at
64,888 ns against its stored 40,900. That is a wisdom-quality note for the
pre-release sweep, NOT a reason to re-race during development.

**What it is not.** Proof of no regression. It cannot see 3–5% drift, and it is not a
re-race — no verdict is banked from it, so it does not violate the racing-budget rule.

---

## 7. What this harness cannot prove

Stated plainly, because the residual risk is being accepted deliberately.

**Sub-threshold performance.** Anything under the same-run control spread is invisible.
Note that §2.6's repair *increases* this risk slightly: making the counters and tunables
`extern` costs the inliner some visibility at those sites. Real, small, knowingly accepted.

**Calibrate-on-miss.** Replay proves "the same decisions, given this frozen store". It
proves nothing about behaviour on a **miss** — the racer machinery is clock-bearing and has
no differential test. The protocol census records the *parameters*; it does not exercise
the *comparison*. A refactor that breaks calibrate-on-miss ships green and surfaces only
when someone adds a new N or deletes a cell.

**MT beyond one configuration.** Bitwise equality plus a strictly-increasing counter proves
threads did work and the answer matched — once, at T=8, on this machine, under this
scheduler. It does not prove the absence of a benign-looking race at other thread counts.

**Wisdom correctness.** Replay proves the store is *read* the same way. It says nothing
about whether the banked verdicts are right.
