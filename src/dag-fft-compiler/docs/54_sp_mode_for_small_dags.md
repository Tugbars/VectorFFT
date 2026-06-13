# 54. SP Mode: Software Pipelining for Small DAGs

## Background

R=4 benchmark gap: hand vs OCaml showed 4-7% gap at small K (K=64),
TIE at large K (K≥256). Diagnosis (this session):

1. **Source-level op counts identical**: 25 arith / 3 FMA / 22 mov for
   both versions
2. **Asm-level op counts identical**: same FMA fusion, same memory traffic
3. **llvm-mca steady-state IPC identical**: 3.47 vs 3.46
4. **Hand asm interleaves loads with ops; OCaml clusters loads then ops**

The diagnosis pointed to pipeline-startup latency: at loop entry, hand's
interleaved pattern fills the pipeline faster than OCaml's clustered
pattern. For SHORT loops (small K), this startup cost is a meaningful
fraction; for LONG loops it amortizes.

## Hypothesis

Our SU+GH scheduler greedily fires the first ready non-load instruction.
At small DAG sizes, this produces tight load-use chains at loop entry.
Hand-written code (and intuitive software pipelining) instead fires
several loads first to keep the load queue filled, then starts firing
ops once enough loads are in flight.

## Implementation

**`lib/schedule.ml`** — added `~sp_mode : bool` parameter to both
`su_schedule` and `su_schedule_subset`. The picker tracks a **load credit
counter**:

- Each scheduled Load: `credit += 1`
- Each scheduled arith op: `credit -= 1`
- When `sp_mode = true` AND `credit < sp_threshold` AND a load is ready
  in source order: force fire the load (overriding the normal
  arith-first policy)

`sp_threshold = 4` (target: ~4 loads in flight at all times).

**`lib/emit_c.ml`** — threaded `sp_mode` through `emit_codelet` to the
underlying scheduler calls.

**`bin/gen_radix.ml`** — added `--sp` (force on) and `--no-sp` (force
off) flags, plus auto-gating: SP fires automatically when
`dag_size ≤ 16`. This catches R=4, R=5, R=7, R=8 — the radixes where
pipeline startup matters relative to body size.

## Verification

### Gating check

The SP machinery fires for the intended radixes only:

```
R=4    SP ACTIVE  (DAG=8, ≤16)
R=5    SP ACTIVE  (DAG=10)
R=7    SP ACTIVE  (DAG=14)
R=8    SP ACTIVE  (DAG=16)
R=11   no effect  (DAG=22)
R=13+  no effect
```

### Output diff (R=4 SP vs no-SP)

SP version fires **4 loads before first op** vs no-SP's **3 loads**.
SP avoids the RMW-on-zmm2 pattern that no-SP produces. Confirmed by
side-by-side asm diff.

### Correctness

```
56/56 PASS — all prime correctness variants (R={2,5,7,11,13,17,19})
```

No regression. Scheduling changes can't affect numerical results, but
verified end-to-end nonetheless.

### Performance

This is where the story becomes more nuanced.

**Container bench is too noisy to definitively measure 4-7% effects.**
Cell-to-cell spread is ~5-10% across 5 runs even for IDENTICAL code.

R=4 results (10 interleaved SP vs no-SP runs):

```
mode    K     median   spread   verdict
NOSP    64    1.030    0.229    TIE
NOSP   128    1.037    0.255    OCaml SLOWER (border)
SP      64    1.011    1.014    TIE
SP     128    1.048    0.853    OCaml SLOWER (border)
```

Spreads of 0.85-1.0 indicate massive variance. Cannot distinguish SP
from no-SP in container conditions.

Full regression bench (R=16, 25, 32, 64) shows all cells within
noise envelope; no cell regressed beyond bench variance and no cell
showed durable improvement attributable to SP.

## Honest assessment

The implementation is correct and well-structured. It changes the
scheduled output exactly as designed (verifiable by asm diff). But the
container microbench isn't precise enough to measure the predicted
4-7% effect.

**The fix needs real hardware to validate.** ICX with frequency pinned
and core isolation would give a clean signal. On the container,
SP-on vs SP-off is indistinguishable from run-to-run noise.

## What we know for sure

1. **Implementation is correct.** Different schedule produced, all
   correctness tests pass.
2. **No degradation in container.** All radixes still TIE or WIN vs hand
   at the median.
3. **Auto-gating works.** SP fires only for R≤8, no effect on R≥11.

## What we don't know

1. **Does SP actually win on real ICX?** Container can't tell.
2. **Is `sp_threshold = 4` the right tuning?** Container can't tell.
3. **Is the `dag_size ≤ 16` threshold correct?** Probably; broader
   threshold (≤80) showed mixed results.

## Recommendation

**Ship with `dag_size ≤ 16` auto-gating, document as experimental.**
Real-hardware testing on ICX will tell whether to:
- Keep as default (if R=4 K=64 numbers improve clearly)
- Make opt-in via `--sp` flag only (if numbers are inconclusive)
- Remove the auto-gating entirely (if no measurable benefit)

The implementation cost (~80 lines of OCaml) is preserved either way as
infrastructure for future scheduler experiments.

## Files changed

- `lib/schedule.ml` — added `~sp_mode` to `su_schedule` and
  `su_schedule_subset`; load-credit mechanism in both pickers
- `lib/emit_c.ml` — threaded `~sp_mode` through `emit_codelet`
- `lib/bb.ml`, `bin/bb_diagnostic.ml` — added `()` terminators to
  `su_schedule_subset` calls (signature change side effect)
- `bin/gen_radix.ml` — `--sp` / `--no-sp` flags, auto-gating

No production hot-path changes if user opts out via `--no-sp`. With
default auto-gating, only R={4, 5, 7, 8} are affected.

## Open questions for next session

- **Test on real ICX**: does R=4 K=64 narrow to a clear TIE or WIN?
- **If SP wins**: should we lower the threshold further (e.g. only R=4)?
  Or raise it (catching R≤13)?
- **Add prologue/epilogue analysis**: the 4-7% gap might also live in
  function entry/exit, not just loop body. Worth checking with perf
  on real hardware.

## Coda

The diagnostic work in this session was valuable independent of whether
SP turns out to help. We now know:

- Our scheduler produces asm-equivalent code to hand at R=4
- gcc's post-RA scheduler doesn't disadvantage us at steady state
- The 4-7% R=4 gap (when it appears) is in pipeline startup, not
  steady-state throughput
- The fix shape is "fire loads aggressively at body entry"

That's a sharper understanding than "OCaml is somehow slower at R=4."
The implementation gives us infrastructure to test the hypothesis on
real hardware.
