# 09 — Decision rationale

ADR-style record of non-obvious decisions in the cost model. Each entry
captures *what we considered, what we picked, and why* — so future
maintainers don't re-litigate already-settled trade-offs from an
incomplete view of the alternatives.

Conventions:
- **Status: ACTIVE** = decision is current; reverting requires a real
  reason
- **Status: SUPERSEDED** = preserved for historical context; the
  superseding decision has its own ADR
- Each entry should be self-contained — a future reader shouldn't have
  to read three other docs to understand it

---

## ADR-001 — Remove the n_decls-based spill penalty

**Status**: ACTIVE
**Date**: 2026-05-03

### Context

An earlier version of `_radix_butterfly_cost_variant` added a penalty
proportional to register pressure:

```c
int spills = max(0, n_decls - reg_budget);   /* reg_budget = 16 on AVX2 */
return (ops + 2.0 * spills) / SIMD_width;    /* 2 work units per spill */
```

The hypothesis was that codelets declaring more SIMD vectors than the
register file holds would generate spill/reload traffic at runtime.

### What we considered

1. **Keep the penalty as written** (`+2 * spills`). Status quo.
2. **Reduce the penalty multiplier** from 2.0 to 0.5 with a higher
   threshold (`max(0, n_decls - 2*reg_budget)`).
3. **Remove the penalty entirely**.

### What we picked

Option 3: removed the penalty. CPE-based cost lookup makes the spill
penalty redundant — if a codelet really spills, that shows up in
measured cycles.

### Why

Bench data on N=64 K=256:
- Cost model with spill penalty picked `4×2×2×2×2` over `8×8`.
- Real bench: `8×8` was actually faster.

Investigation: R=8 has `n_decls = 94`. Penalty added `(94-16) * 2 / 4
= 39` to the base cost of 22.5 → effective cost 61.5, almost 3× the
real number.

But VTune showed R=8 codelets *do not actually spill on AVX2*. ICX
splits live ranges aggressively; most of the 94 declared vectors have
short liveness windows that don't overlap. n_decls is a peak-liveness
*proxy*, not an actual spill count.

The over-penalization was bad for any codelet > 16 declarations, which
is most of them past R=4. Removing it gave us a clean +0.36×
improvement in mean estimate/wisdom ratio (1.69 → 1.33). The CPE
table picks up real spill costs (when they exist) inside its measured
numbers.

### Trade-offs

- **Loss**: We no longer penalize a hypothetical future ISA where
  spills are catastrophic but the CPE table hasn't been measured. In
  that hypothetical, ops/SIMD fallback still applies.
- **Gain**: The model picks correctly on every existing radix.

---

## ADR-002 — Measure at K=256 only, not a K-grid

**Status**: ACTIVE
**Date**: 2026-04-30

### Context

The CPE table contains one cycles-per-butterfly value per (R, variant,
ISA). The natural extension is to measure at multiple K and store a
2D table indexed by (R, K), or fit a small model `cycles(K)` per radix.

### What we considered

1. **Single K (current)**. One number per (R, variant, ISA).
2. **K-grid**: measure at K ∈ {4, 32, 256, 1024}, store a 2D table.
3. **Parametric model**: fit `cycles(K) = a + b*K + c*K²` per radix.

### What we picked

Option 1: single K=256 baseline. Adjust for non-baseline K via
`cache_factor` in the score function.

### Why

The bench shows the model picks well at K=4 and K=1024 even though
the CPE table is K=256-only. The reason: the per-butterfly compute
*structure* is K-invariant (same instruction stream), and the
K-dependent cost (working-set fit, stride pressure) is captured by
the `cache_factor` step function in `stride_score_factorization`.

Concrete bench evidence (N=64):
- K=256: estimate picks `4×4×4`, matches wisdom (1.0× tie)
- K=1024: estimate picks `4×4×4`, matches wisdom (1.0× tie)

Both inferred from the same K=256 CPE numbers. The model's K-handling
works.

A K-grid would 4× the calibration time and introduce K-axis
interpolation issues (what's `cycles(K=128)` between measured K=32 and
K=256?). The single baseline is a strict subset of "good enough" with
zero added complexity.

### Trade-offs

- **Where it breaks**: Very small K (K<8) where loop overhead
  dominates. Cost model under-costs K=4 stages relative to reality.
- **Where it could break**: Very large K (K>4096) where cross-stage
  cache misses compound in ways the step function doesn't model. Not
  in the bench today, so uncalibrated.

Both are noted in [00_thesis.md](00_thesis.md) §4 as v1.x improvement
candidates — neither is a v1.0 blocker.

---

## ADR-003 — No runtime probe at vfft_init() in v1.0

**Status**: ACTIVE
**Date**: 2026-05-03

### Context

The cost model needs CPE numbers in `radix_cpe.h`. Two ways to
populate them:

A. **Build-time**: codelet maintainer runs `measure_cpe.c` (manually
   or via orchestrator), commits the resulting header. Numbers
   reflect the calibration host.

B. **Runtime**: `vfft_init()` detects an empty/missing CPE table and
   probes ~50ms on first call. Numbers reflect the user's host.

### What we considered

1. **(A) only** — ship one calibration host's numbers, accept
   cross-host inaccuracy.
2. **(B) only** — every process pays a 50ms probe at init, headers
   are not host-specific.
3. **(A) primary, (B) fallback** — ship calibrated numbers, run probe
   only if the table is empty. User-transparent.

### What we picked

Option 1 for v1.0. Option 3 is a safety-net feature for v1.x.

### Why

The codelet generation pipeline (`orchestrator.py`) already requires
running on the target host as part of the build process. CPE
measurement piggybacks on that step (`--phase cpe_measure`). End
users who consume pre-built codelets get the calibration host's
numbers, which are good enough as long as the host is microarchitecturally
similar (Raptor Lake → Sapphire Rapids: probably fine; → Zen 4: worth
re-measuring).

The runtime probe path is **architectural insurance**, not a feature
end users need. If they're running on the same CPU family the
codelets were calibrated for, the shipped numbers are correct. If
not, they should re-run `cpe_measure` for their host (a 15-second
operation). The runtime probe is a fallback for the "user built from
source on a different machine and forgot to regenerate the header"
case.

### Trade-offs

- **Loss**: A user on a wildly different CPU (different family,
  different uarch) gets the calibration host's CPE numbers, which
  may mis-rank radixes for them. ESTIMATE quality degrades; fallback
  is to use MEASURE.
- **Gain**: Zero first-run cost, no surprise 50ms init delay,
  deterministic behavior across runs of the same process.

When v1.x adds the runtime probe, the lookup hierarchy becomes
`runtime → compile-time → ops/SIMD fallback`, with the runtime layer
populated from a one-off probe at first plan creation.

---

## ADR-004 — Variance check at 5% CV in measure_cpe

**Status**: ACTIVE
**Date**: 2026-05-03

### Context

`measure_cpe.c` runs each codelet for 51 batches and computes the
coefficient of variation across batch medians. If CV exceeds a
threshold, the tool refuses to overwrite `radix_cpe.h` (unless
`--force`).

What threshold?

### What we considered

| Threshold | Effect |
|-----------|--------|
| 1% | Refuses everything except a perfectly idle calibration host |
| 5% | Admits good calibration runs; refuses noisy hosts |
| 10% | Admits most calibration runs; admits some noisy ones |
| 20% | Admits almost everything |

### What we picked

5%. The orchestrator's `cpe_measure` phase commits headers under this
threshold; `--force` is a development-only escape valve.

### Why

Empirical observation across many runs:

- **Calibration-grade host** (P-core pinned, performance plan,
  no other load): typically 1–3% CV. 5% is comfortably above this
  band.
- **Consumer PC at idle** (the development machine): typically 5–15%
  CV when the system is quiet, jumping to 30–90% during background
  activity (chrome, indexer, etc).
- **Calibration-grade host with one chrome tab open**: 8–20% CV.

5% is the band edge that admits clean calibration runs and refuses
everything else. Below 5%, we'd refuse legitimate runs; above 10%,
we'd admit measurements that materially mis-rank radixes.

The threshold is documented at the top of `measure_cpe.c` and in the
`radix_cpe.h` fingerprint comment block. PRs that touch the header
should have `Max CV < 5%` in the comment block — anything else means
the calibration host wasn't quiet, and reviewers should ask why.

### Trade-offs

- **Loss**: Some legitimate calibration runs get rejected on
  borderline hosts. Workaround: `--force`. Reviewers can spot it
  via the fingerprint and push back.
- **Gain**: The cost-model accuracy floor is enforced at commit time,
  not discovered later by users seeing bad picks.

---

## ADR-005 — Empirical CPE supersedes hand-coded VTune extracts

**Status**: ACTIVE (supersedes pre-2026-05-03 hand-coded table)
**Date**: 2026-05-03

### Context

Pre-`measure_cpe.c`, the per-radix cycle costs were hand-coded into
`factorizer.h` as a switch table, derived from VTune profile docs in
`docs/vtune-profiles/`:

```c
case  4: return  2.64;  /* measured: 118.6 ns × 5.71 GHz / 256 */
case  8: return  9.71;  /* measured: 437.8 ns × 5.68 GHz / 256 */
/* ... */
```

This worked — it was what got us to mean ratio 1.19× — but had three
serious problems.

### What we considered

1. **Keep the hand-coded table.** Update manually when codelets change.
2. **Extend the existing `bench_codelet.c` harness** (in
   `src/stride-fft/bench/`) to emit a header. But that harness targets
   the OLD `stride-fft/codelets/` tree, not the new core's
   `vectorfft_tune/generated/`.
3. **Build a new harness specifically for the new core** that times
   codelets through the registry (`stride_registry_t`) and emits a
   header. — Picked.

### What we picked

Option 3. Wrote `tools/radix_profile/measure_cpe.c` from scratch as a
new-core-aware timing harness.

### Why

Hand-coded numbers fail in three ways:

1. **They go stale silently.** When a codelet is regenerated with new
   intrinsics or different scheduling, the hand-coded number doesn't
   update. The cost model keeps using last quarter's measurements
   without anyone noticing.

2. **They're a maintenance burden.** Adding a new radix means updating
   the switch table. Porting to a new ISA means duplicating it.
   Forking for AVX-512 means a parallel switch table that drifts
   from AVX2.

3. **They tie the build to one host's profile.** The VTune docs were
   from one specific i9-14900KF run. Anyone building VectorFFT on
   different hardware would inherit those exact numbers, including
   their idiosyncrasies.

Option 2 (extend `bench_codelet.c`) was tempting because the timing
infrastructure exists — but it targets the wrong codelet tree. The
new core consumes codelets from `vectorfft_tune/generated/`, not
`stride-fft/codelets/`. Extending the old harness would have us
measuring numbers for the wrong codelets.

Option 3 made the CPE numbers a build artifact — auto-generated by
the same pipeline that generates `radix_profile.h`, host-specific
by design, with a fingerprint comment block in the output. Anyone
regenerating codelets also regenerates CPE; anyone on different
hardware can re-measure cleanly.

### Trade-offs

- **Loss**: We lost the per-radix VTune deep-dive context that the
  hand-coded comments preserved. Mitigation: the
  `docs/vtune-profiles/` markdown files retain the analysis; the
  CPE table now lives separately and just records numbers.
- **Gain**: The numbers are auto-generated, reproducible, host-aware,
  and updated whenever the codelets change.

---

## ADR-006 — log3 takes precedence over t1s in variant selection

**Status**: ACTIVE
**Date**: 2026-05-03

### Context

When both predicates fire on a stage (`stride_prefer_dit_log3` AND
`stride_prefer_t1s`), which variant does the cost model assume the
executor will use?

### What we considered

1. log3 first, t1s second
2. t1s first, log3 second
3. min(cyc_log3, cyc_t1s)

### What we picked

Option 1: log3 first.

### Why

This isn't a cost-model choice — it's mirroring `_stride_build_plan`.
That function (`src/core/planner.h`) attaches log3 first and sets
`stage_skip_t1s = 1` so t1s isn't overlaid. Order:

```c
if (want_log3) {
    t1f[s] = reg->t1_fwd_log3[R];
    stage_skip_t1s[s] = 1;
}
else if (want_buf) {
    t1f[s] = reg->t1_buf_fwd[R];
    stage_skip_t1s[s] = 1;
}
else {
    t1f[s] = reg->t1_fwd[R];
}
/* later: if !stage_skip_t1s && prefer_t1s, attach t1s overlay */
```

The cost model has to use the same precedence. Otherwise the model
predicts t1s cost but the executor runs log3 — drift between cost
prediction and runtime behavior.

This is a structural mirror, not a free choice. Option 2 would make
the model wrong on cells where log3 actually wins. Option 3 would
predict cycles cheaper than what runs (best of both, but the executor
only picks one).

In practice, the predicates are designed to be **mutually
exclusive** — `prefer_log3` and `prefer_t1s` rarely both fire. The
ordering matters mostly for safety: if a future calibrator emits
non-exclusive predicates, the cost model still doesn't drift.

---

## ADR-007 — Auto-generated headers live in src/core/generated/, not tools/

**Status**: ACTIVE (refactor on 2026-05-03)
**Date**: 2026-05-03

### Context

`radix_profile.h` and `radix_cpe.h` are auto-generated by tools in
`tools/radix_profile/`. Where should the **headers themselves** live?

### What we considered

1. Headers in `tools/radix_profile/` next to their generators. Simple
   and locality-preserving.
2. Headers in `src/core/generated/`, generators stay in `tools/`.

### What we picked

Option 2. Refactor done on 2026-05-03.

### Why

Auto-generated headers are part of the **library's compile-time
surface** — `factorizer.h` cannot compile without them. They are as
load-bearing as `executor.h`. Their location should reflect that.

Putting them in `tools/` introduced three problems:

1. **Weird include paths**: `build.py` had to add
   `-Itools/radix_profile` to every compilation unit's include path.
   The IDE's clangd indexer didn't always see the flag, leading to
   recurring "stale diagnostic" noise during the cost-model work.

2. **Header pattern violation**: every other auto-generated header in
   the project lives next to its consumer (e.g., codelets in
   `vectorfft_tune/generated/`). Two different conventions for two
   similar artifacts.

3. **Discoverability**: "where is `stride_radix_cpe_avx2` defined?" —
   `find src -name '*.h'` should answer that. With the headers in
   `tools/`, they wouldn't.

### Trade-offs

- **Loss**: Slight cognitive split — generator in one folder, output
  in another. Mitigated by `tools/radix_profile/README.md` linking
  to the output paths.
- **Gain**: No more `-Itools/` include hops. IDE happy. Headers grep'able from `src/`.

The CSV side-products of `extract.py` (`profile_avx2.csv`,
`profile_avx512.csv`) stay in `tools/` as developer artifacts —
they're not consumed by the build.

---

## ADR-008 — Variance check refuses commit, doesn't fall back

**Status**: ACTIVE
**Date**: 2026-05-03

### Context

When `measure_cpe` exceeds the 5% CV threshold, what should it do?

### What we considered

1. **Refuse the entire emit** (current behavior) — exit non-zero,
   leave the existing header untouched.
2. **Emit partial header** — write only the codelets whose individual
   CV passed; leave others as zero (cost model falls back to
   ops/SIMD).
3. **Emit anyway with a warning comment** — write the noisy numbers
   but make it visually obvious in the file.

### What we picked

Option 1. Whole-or-nothing emit.

### Why

Cost models are *relative* — what matters is the ranking of radixes
against each other. A noisy R=8 number throws off comparisons against
R=4 and R=16, even if R=4 and R=16 themselves were measured cleanly.

Option 2 (partial emit) creates worse problems than it solves. The
fallback path uses ops/SIMD, which has very different numerical
characteristics from CPE (typically 5–10× higher absolute magnitude).
Mixing CPE and ops/SIMD in the same comparison gives nonsense
rankings.

Option 3 (noisy emit with comment) was tempting — at least the
fingerprint shows the variance is bad — but humans don't reliably
check fingerprints. The discipline has to be enforced at emit time.

The whole-or-nothing approach has a sharp edge: a noisy host
*completely fails* the regen. That's intentional. It forces the
operator to either (a) get the host into a quieter state, or (b)
explicitly opt in via `--force` and document the override. Both are
preferable to silently committing noisy numbers.

---

## ADR-009 — Per-radix bench_codelet (stride-fft) deprecated for cost-model use

**Status**: ACTIVE (decoupling from 2026-05-03)
**Date**: 2026-05-03

### Context

`src/stride-fft/bench/bench_codelet.c` exists and emits VTune-grade
codelet timings. We considered using its output for the cost model.

### What we considered

1. **Use bench_codelet** — its harness has been stable for months,
   used to produce `docs/vtune-profiles/`. Just parse its output.
2. **New harness (`measure_cpe.c`)** — duplicates timing logic but
   targets the right codelet tree.

### What we picked

Option 2. bench_codelet stays for VTune correlation but is no longer
load-bearing for the cost model.

### Why

bench_codelet links against `src/stride-fft/codelets/`, the old
production codelet tree. The new core (`src/core/`) consumes codelets
from `src/vectorfft_tune/generated/` via the registry. Using
bench_codelet's output would mean the cost model is calibrated against
**the wrong codelets**.

This was actually invisible for the hand-coded table era because the
two codelet trees produce nearly identical numbers (same algorithms,
same intrinsics) — but as soon as we wanted regeneration to be
automated, the discrepancy mattered. `measure_cpe` always sees what
the new core actually runs.

### Trade-offs

- **Loss**: Two timing harnesses now exist. Need to keep them
  reconciled if both are used (e.g., for VTune correlation).
- **Gain**: Cost model is calibrated against the executor's actual
  codelets. No silent drift.

---

## See also

- [00_thesis.md](00_thesis.md) — the conceptual framework these
  decisions live inside
- [03_dynamic_cpe.md](03_dynamic_cpe.md) — where the variance-check
  threshold and emit policy are implemented
- [05_variant_selection.md](05_variant_selection.md) — where the
  log3-before-t1s precedence is implemented
- [06_validation.md](06_validation.md) — the bench data underlying
  the empirical claims in these ADRs
