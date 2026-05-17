# prototype-core wisdom calibrator

Single-cell calibrator that runs multiple planners (patient, dp, estimate),
picks the fastest measured plan, and emits a wisdom line in production
format suitable for appending to `src/prototype/generated/spike_wisdom.txt`.

## What it does

For each `(N, K)` cell:

1. Run one or more planners (controlled by `--mode`):
   - `patient` — `exhaustive_patient.h`: enumerates all factorizations, benches
     each with warmups + 7 trials + inter-trial pacing, returns best. Gold
     standard but slow (~minutes at high N).
   - `dp` — `dp_planner.h`: recursive memoized planner. Picks one
     factorization in seconds.
   - `estimate` — `estimate_plan.h`: V4 cost model. Picks one factorization
     in milliseconds (no benchmarking during planning).
   - `best` (default) — runs all three, picks the fastest measured ns.

2. For each chosen planner, build the plan and bench under a consistent
   harness (10 warmups, 7 trials, 100ms inter-trial pacing) so results
   across planners are apples-to-apples.

3. Pick the winner (lowest measured ns) and emit a wisdom line on stdout
   in production format:
   ```
   N K nf factors... best_ns 0 0 0 0 use_dif_forward variants...
   ```

## Orientation (DIT vs DIF)

The `--orient` flag controls which orientation(s) to try:
- `dit` (default) — DIT-only (current production path)
- `dif` — DIF-only (code paths plumbed but not the default)
- `both` — try both orientations, pick winner (doubles wall time)

DIT/DIF cannot be mixed within one plan (would require a permutation between
stages, killing latency). The DIF code path is fully plumbed through patient,
DP, and estimate but is disabled by default until DIF wisdom is validated.

## Usage

```
# Build
bash src/prototype-calibrator/build_calibrate.sh

# Single cell
src/prototype/build_tuned/calibrate.exe 1024 4 --mode best --orient dit

# Via PowerShell orchestrator (P-core pinned, priority=High, auto-append to wisdom):
powershell -File src/prototype-calibrator/run_calibrate.ps1 -N 1024 -K 4
powershell -File src/prototype-calibrator/run_calibrate.ps1 -Cells "128:4,256:4,512:4"

# Regen plan_executors.h after appending new entries:
wsl --cd /mnt/c/tmp bash regen_executor.sh
```

## Methodology notes

- **CPU pinning**: orchestrator pins to CPU 2 (first clean P-core; CPU 0/1
  handle interrupts on consumer Intel). Matches production's `calibrate.py`.
- **Power plan**: must be High Performance or Ultimate Performance.
  Orchestrator warns if not.
- **Variance**: bench is best-of-7-trials with 100ms inter-trial pacing.
  Patient adds inter-candidate pacing + top-5 rebench. Single-trial timing
  is unreliable (see `M_PROJECT.md §9.1`).

## Output format

Production wisdom format (`vfft_wisdom_tuned.txt` shape):

```
N K nf factors[nf] best_ns use_blocked split_stage block_groups use_dif_forward variants[nf]
```

- `use_blocked`, `split_stage`, `block_groups` are always 0 (prototype-core
  doesn't support blocked execution).
- `use_dif_forward` reflects the winning orientation.
- Variants: DIT writes `0` for stage 0 (no-twiddle outer) and `2` for inner
  stages (T1S). DIF writes `0` everywhere (DIF has no T1S codelets).
