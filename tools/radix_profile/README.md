# tools/radix_profile

Generators for the per-radix profile tables consumed by the cost model in
`src/core/factorizer.h`.

## What's here

| File                   | Kind        | Purpose                                       |
|------------------------|-------------|-----------------------------------------------|
| `extract.py`           | tool        | Static analysis: parses generated codelets, counts intrinsics, emits `radix_profile.h`. |
| `measure_cpe.c`        | tool        | Dynamic measurement: times each registered codelet variant at K=256, emits `radix_cpe.h`. |
| `profile_avx2.csv`     | dev artefact | Per-radix per-variant op counts, AVX2.       |
| `profile_avx512.csv`   | dev artefact | Per-radix per-variant op counts, AVX-512.    |

## Where the headers go

The auto-generated headers consumed by the library are NOT in this folder.
They live next to the consumer:

- `src/core/generated/radix_profile.h` — written by `extract.py`
- `src/core/generated/radix_cpe.h`     — written by `measure_cpe.c`

Treat those files as build artifacts. Do not hand-edit them — regenerate
through the workflow below.

## Regeneration workflow

### `radix_profile.h` (op counts, deterministic)

Static parse, runs anywhere with Python:

```
python tools/radix_profile/extract.py
```

Output is identical run-to-run on the same codelet tree. Re-run after any
codelet regeneration (`gen_radix*.py`) so the profile reflects the latest
intrinsics.

### `radix_cpe.h` (cycles per butterfly, host-specific)

Dynamic measurement; requires a calibration-grade host:

```
# 1. Compile the harness (uses the same build script as everything else)
python build_tuned/build.py --src tools/radix_profile/measure_cpe.c

# 2. Run it. By default this enforces a 5% coefficient-of-variation
#    threshold across BENCH_N_RUNS=21 runs — if the host is too noisy,
#    the tool refuses to overwrite the header and exits non-zero.
tools/radix_profile/measure_cpe.exe
```

Recommended host state:
- single physical core, pinned (`taskset -c 0` / `SetProcessAffinityMask`)
- High Performance / `performance` governor active
- no other significant load

For a calibration-grade run that handles affinity + power-plan switching
automatically, route through the orchestrator's `cpe_measure` phase
(orchestrator wiring TODO — see step #2 in the v1.0 plan).

### Bypass flags for `measure_cpe.exe`

| Flag         | Effect                                                |
|--------------|-------------------------------------------------------|
| `--force`    | Write the header even if CV > 5%. Use only for development on a noisy machine; never commit a forced header. |
| `--no-emit`  | Print results without writing the header (dev iteration). |
| `--verbose`  | Print cycles/butterfly summary at the end.            |
| `--output=…` | Custom output path (default `src/core/generated/radix_cpe.h`). |

## Why this split

`radix_profile.h` is a deterministic byproduct of source code — a
re-extraction yields the same numbers on any host, so it can be regenerated
freely and the result is portable.

`radix_cpe.h` is empirical — its numbers describe the calibration host's
microarchitecture. The repo ships one host's numbers (the calibration
machine) as a baseline. Users on a materially different CPU can re-measure
and either replace the file or override at runtime via the
`vfft_calibrate_cpe()` API (TODO — step #3).

The fingerprint comment block at the top of every committed `radix_cpe.h`
documents which machine produced it, so reviewers can spot stale or
wrong-platform commits.
