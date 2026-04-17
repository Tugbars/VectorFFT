# VectorFFT Codelet Bench (R=16)

Per-chip calibration tool. Runs on every target chip class, produces a
chip-specific selector that the planner consumes.

## What this does

Your generator emits many codelet variants — plain `ct_t1_dit`, buffered
variants at different tile sizes with temporal/stream drains, log3,
scalar-broadcast. Which variant wins at which `(ios, me)` depends on the
target microarchitecture. This bench measures all variants across a grid
of `(ios, me)` points on the current machine and emits a C selector:

```c
codelet_r16_t1_fn fn = codelet_select_r16_fwd(isa, ios, me);
```

The selector is a compile-time lookup. No runtime search, no JSON parsing.

## Design principles

1. **Don't prune on intuition.** We bench the full Cartesian first. Pruning
   rules (e.g. "log3 never wins when K<128") come from evidence across
   multiple chips, not armchair reasoning on one.

2. **Multi-region generation.** We don't pick "one tile size per chip."
   If `buf_tile128` wins at K≥512 and `t1s` wins at K<128 and `log3`
   wins at padded strides, we emit all three and the selector dispatches.

3. **µarch decides codelets; planner decides factorizations.** This bench
   produces a codelet-selection header. The planner/wisdom system
   (existing) handles factorization choice separately. Two layers, each
   doing what they do best.

4. **User owns noise.** We do median of 21 samples. If you want
   ultra-reliable selections, you're responsible for quiet machine state,
   core pinning, disabling turbo, etc.

## Pipeline

```
generate-all.bat (or equivalent):
    python3 bench/bench_codelets.py     # generate + build + run → measurements.json
    python3 bench/select_codelets.py    # pick winners → selection.json
    python3 bench/emit_selector.py      # write include/codelet_select_r16.h
```

Each phase writes a JSON artifact and reads the previous one. You can
tweak any later phase and re-run without repeating the bench (e.g. change
tie threshold, re-run `select_codelets.py` only).

## Outputs

- `measurements.json` — full raw bench data (~1 KB per measurement × 792 = ~30 KB)
- `selection.json` — winners per region + list of kept candidates
- `include/<winner>.h` — source for each winning codelet (only winners ship)
- `include/codelet_select_r16.h` — the C selector API
- `codelet_select_r16_report.md` — human-readable summary

## Candidate matrix (R=16)

22 candidates across both ISAs. See `candidates.py` for authoritative list.

| Family | Knobs | Variants per ISA |
|--------|-------|-------------------|
| ct_t1_dit      | none                          | 1 |
| ct_t1_dit_log3 | none                          | 1 |
| ct_t1s_dit     | none                          | 1 |
| ct_t1_buf_dit  | tile ∈ {16,32,64,128} × drain ∈ {temporal, stream} | 8 |

Two directions (fwd + bwd), so effectively 44 codelet-direction pairs per ISA.

## Sweep shape

`me ∈ {64, 128, 256, 512, 1024, 2048}`
For each `me`, `ios ∈ {me, me+8, me+64}`.

Total: 18 `(ios, me)` points × 22 candidates × 2 directions = **792 measurements**.

Extend `candidates.py` to widen or narrow the sweep.

## Runtime

On an AVX-512 SPR container with one thread: **~2-3 minutes** for full sweep
(generate + build + run). Windows + MSVC should be similar. Scaling linearly
if you extend the sweep.

## Caveats

- **Selector is machine-specific.** Do not commit generated `include/*.h`
  files to a repo shared between different chip classes. Regenerate on
  each target.
- **Bench runs single-threaded by design.** Threading decisions live in
  the planner, not the codelet bench.
- **Off-grid calls fall back to nearest measured point** (Euclidean in
  log2 space). For best results, bench at `(ios, me)` points close to
  what your actual workloads hit.
- **No AVX-512 skip at measurement.** On hosts without AVX-512, the
  bench silently skips AVX-512 candidates at runtime (harness checks
  `__builtin_cpu_supports`). The output `measurements.json` will have
  `"avx512_available": false`.
- **Tie threshold (default 2%)** trades off between absolute best-ns
  and selection stability. Increase for more stable selections across
  re-runs. Decrease for hotter-and-cooler sensitivity.

## Extending to other radixes

Copy the R=16 pattern:

1. Create `candidates.py` entries for R=N families and knobs
2. Extend the generator-CLI logic in the harness template if needed
3. Duplicate `emit_selector.py` as `emit_selector_rN.py` (the function
   signature class might differ)

Eventually the three scripts should take `--radix N` and discover
everything from a shared candidates module indexed by N. v1 doesn't do
that; v2 should.
