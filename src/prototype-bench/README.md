# prototype-bench

vfft prototype-core vs MKL bench harness. Loads `spike_wisdom.txt`,
builds each entry's plan with `vfft_proto_plan_create_ex` (uses the
wisdom-recorded factorization + variants + orientation), benches forward
execution against MKL on identical input data + layout.

Companion to:
- `src/prototype-calibrator/` — produces / refreshes `spike_wisdom.txt`
- `src/prototype/bin/emit_executor_h.ml` — emits Tier 1 executors from wisdom

## Build

```
bash src/prototype-bench/build_bench_1d_vs_mkl.sh
```

Reuses the codelet `.o` cache produced by the demo build scripts. Run
`build_demo_dp_one_cell.sh` once first if the cache is cold.

## Usage

```
# Bench every wisdom entry
src/prototype/build_tuned/bench_1d_vs_mkl

# Filter to specific cells
src/prototype/build_tuned/bench_1d_vs_mkl --cells "131072:4,1024:128"

# Write CSV (recommended path: src/prototype-bench/results/)
src/prototype/build_tuned/bench_1d_vs_mkl \
  --csv src/prototype-bench/results/bench_1d_vs_mkl.csv

# Custom wisdom path (default: src/prototype/generated/spike_wisdom.txt)
src/prototype/build_tuned/bench_1d_vs_mkl --wisdom path/to/wisdom.txt
```

## Output

Console: per-cell row with vfft ns, mkl ns, vs-MKL ratio, GFLOPS.

CSV columns:
```
N,K,orient,factors,vfft_ns,mkl_ns,ratio_mkl_over_vfft,vfft_gflops,mkl_gflops
```

## Methodology

- 10 warmups + best-of-5 trials per engine (same harness as production
  `bench_1d_vs_mkl.c`)
- `mkl_set_num_threads(1)` — MKL forced to single-threaded for apples-
  to-apples vs prototype-core (no MT yet)
- Cache-bust between vfft and MKL benches (32 MB junk sweep) so each
  engine measures from a clean cache state
- Identical input buffer per cell (seeded `srand(42 + N)`)

For thermal-stable / pinned measurement, run from the calibrator
orchestrator's pinning mode or invoke via:

```
$proc = Start-Process bench_1d_vs_mkl ...
$proc.ProcessorAffinity = 0x4   # CPU 2 (first clean P-core)
$proc.PriorityClass     = 'High'
```

## Known gaps

- Single-radix cells (N≤64) show bimodal timing (~140ns vs ~240ns) on
  Raptor Lake. Cause not yet isolated — could be TLB warmth or branch
  predictor state interaction with the straight-line codelet body.
- Production codelets are ~1.5× faster than prototype's defaults at
  R=64 (production has per-radix vectorfft_tune calibration; prototype
  uses OCaml-emitter defaults). Future work: port the codelet
  calibration step.
