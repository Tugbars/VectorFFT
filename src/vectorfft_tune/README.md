# VectorFFT codelet bench + dispatch infrastructure

Radix-generic ATLAS-style autotuning for VectorFFT codelets. Per-chip
calibration emits dispatcher headers that the registry picks up.

## What this does

For a given radix R, measures every codelet variant across a sweep of
`(me, ios)` points on the current machine, then emits per-protocol
dispatcher headers plus a plan-wisdom header:

    vfft_r{R}_t1_dit_dispatch.h       # flat protocol dispatcher
    vfft_r{R}_t1_dit_log3_dispatch.h  # log3 protocol dispatcher
    vfft_r{R}_t1s_dit_dispatch.h      # t1s protocol dispatcher
    vfft_r{R}_plan_wisdom.h           # protocol selection for planner

The generated headers drop into the existing VectorFFT build; the
registry's `_REG_T1(R)` / `_REG_T1_LOG3(R)` / `_REG_T1S(R)` macros
resolve to the dispatcher (which is `static inline` and picks among
its variants based on `(me, ios)`).

## Architecture

One dispatcher per protocol per radix. A **protocol** is a contract
between the planner (which populates the twiddle buffer) and the
codelet (which reads from it):

- **flat**: `W_re` has `(R-1) * me` doubles; codelet reads per-m per-leg.
  Variants: `ct_t1_dit`, `ct_t1_dit_u2`, `ct_t1_dit_log1`.
- **log3**: `W_re` has `me` doubles (w1 only); codelet derives w2, w3.
  Variants: `ct_t1_dit_log3`.
- **t1s**: `W_re` has `R-1` scalars; codelet broadcasts once before the
  m-loop. Requires K-blocked execution at the planner level.
  Variants: `ct_t1s_dit`.

Within each protocol, the bench picks the winning variant per
`(me, ios)`. Across protocols, plan-wisdom emits decision functions
(`radix{R}_prefer_log3(me, ios)` etc) that the planner consults when
building a plan.

## Pipeline

    python common/bench.py --radix-dir radixes/r4 --phase generate
    python common/bench.py --radix-dir radixes/r4 --phase compile
    python common/bench.py --radix-dir radixes/r4 --phase run
    # --> bench_out/r4/measurements.jsonl

    python common/select_and_emit.py --radix-dir radixes/r4
    # --> generated/vfft_r4_*.h

Each phase idempotent. `--phase run` resumes from existing jsonl.

## Output layout

After `--phase all` (or `--phase emit`), the `generated/r{N}/` directory is
the complete, drop-in library for radix N:

```
generated/r8/
├── fft_radix8_avx2.h              codelet bodies for all 8 variants (AVX2)
├── fft_radix8_avx512.h            codelet bodies for all 8 variants (AVX-512)
├── vfft_r8_t1_dit_dispatch_avx2.h per-dispatcher selection wrappers
├── vfft_r8_t1_dit_dispatch_avx512.h
├── vfft_r8_t1_dif_dispatch_avx2.h
├── ...                            (one pair per dispatcher)
├── vfft_r8_plan_wisdom.h          machine-readable winner table
└── vfft_r8_report.md              human-readable results summary
```

To use from a downstream codebase: add `generated/r{N}/` to your include
path and `#include "vfft_r{N}_<dispatcher>_dispatch_<isa>.h"`. The
dispatcher wrapper pulls in the codelet header automatically. Each
variant wrapper is `static inline` so there is nothing to link — the
compiler inlines the winning codelet directly at the call site.

AVX-512 headers are generated on every host (they're just intrinsic
code), but only compiled and benched on hosts where AVX-512 is
actually available. Carrying both sets lets the same `generated/`
tree be used on any Intel chip.

## Portability

### Linux (default)
Container or Ubuntu with `gcc` on PATH. Nothing special.

```
python3 common/bench.py --radix-dir radixes/r4 --phase all
```

### Windows + ICX (Tugbars's production target)

Set up Intel oneAPI environment (ICX + LLD), then:

```
set CC=icx
python common\bench.py --radix-dir radixes\r4 --phase all
```

ICX on Windows uses LLD linker by default. AVX-512 candidates auto-skip
via CPUID at harness runtime (Raptor Lake has AVX-512 fused off).

### Windows + MSVC (fallback)

```
set CC=cl
python common\bench.py --radix-dir radixes\r4 --phase all
```

MSVC only accepts one `/arch:` at a time; infrastructure picks the widest
ISA any candidate requires.

### Override the compiler

`VFFT_CC` takes precedence over `CC`. Absolute paths work:

```
set VFFT_CC=C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icx.exe
```

## Adding a new radix

Create `radixes/rN/` with:

- `gen_radixN.py` — codelet generator. Must expose a `VARIANTS` dict
  mapping variant IDs to `(function_base, protocol)`, plus `function_name(v, isa, dir)` and `protocol(v)` helpers.
  Must support `--isa {avx2|avx512|scalar}` to emit the ISA header.
- `candidates.py` — exposes `RADIX`, `GEN_SCRIPT`, `enumerate_all()`,
  `sweep_grid(variant)`, `function_name(v, isa, dir)`, `protocol(v)`.

The bench infrastructure in `common/` is radix-agnostic — no changes
required.

## Files in bench_out/r{R}/

- `staging/fft_radix{R}_{isa}.h` — generator output per ISA
- `build/vfft_harness_candidates_{isa}.c` — per-ISA candidate tables
- `build/vfft_harness_candidates.c` — aggregator
- `build/harness[.exe]` — measurement binary
- `measurements.jsonl` — raw results, one JSON object per line

## Troubleshooting

**"No C compiler found"**: install gcc/clang/icx, or set `VFFT_CC`.

**AVX-512 candidates measured as "skipped" on Raptor Lake**: expected —
Raptor Lake has AVX-512 fused off. The bench correctly skips them via
`__builtin_cpu_supports("avx512f")` CPUID.

**Duplicate symbol errors at link time**: the infrastructure splits one
TU per ISA to avoid scalar-codelet name collisions. If you hit this,
check that `emit_candidate_table` is producing separate `vfft_harness_candidates_{isa}.c`
files and not a single shared one.

**Windows cp1252 decode errors**: make sure `subprocess` calls go through
`common/bench.py`'s `_run` helper, which forces `PYTHONIOENCODING=utf-8`
on child processes.


## Orchestrator — full-portfolio bench

For a full sweep across all tuned radixes, use `common/orchestrator.py` instead
of invoking `common/bench.py` per-radix by hand:

```
python3 common/orchestrator.py             # runs all discovered radixes
python3 common/orchestrator.py --radix r10,r12   # subset
python3 common/orchestrator.py --skip r25        # all except
python3 common/orchestrator.py --dry-run         # show commands, don't execute
python3 common/orchestrator.py --phase emit      # just emit phase across all
```

**Windows + Intel compiler (icx)**: the shell you launch the orchestrator
from must have `setvars.bat` already active, otherwise the Intel linker
won't find its own runtime (`LNK1104: cannot open file 'libircmt.lib'`).
Running setvars in a different shell and then launching the orchestrator
from a fresh cmd won't work — environment variables like `LIB` aren't
inherited across unrelated shells.

```
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
python common\orchestrator.py
```

**Benchmark hygiene built in:**
- Pre-flight check: CPU governor (Linux) / power plan (Windows) must be
  performance-class; aborts otherwise. `--force` to bypass.
- CPU affinity: all bench runs pinned to one core via `taskset -c N` on
  Linux or `start /AFFINITY` on Windows. Default CPU 2 (first "clean"
  P-core on most consumer Intel); override with `--cpu N` or config file.
- Thermal pacing: 2s sleep between radix runs (adjust via `--pace-seconds N`).
- Sequential execution: measurement phases never parallelize.

**Config file**: `orchestrator.json` at project root sets defaults.
CLI args override config values override builtins. Example:

```json
{
  "cpu": 4,
  "pace_seconds": 3,
  "phase": "all"
}
```

**Outputs**:
- `bench_out/ALL_SUMMARY.md` — cross-radix win-count tables, regime
  transitions, portfolio stats (regenerated from all measurement data
  on disk, not just this invocation)
- `bench_out/rN/orchestrator.log` — per-radix captured output
- `bench_out/orchestrator.log` — overall sweep log

**Other options**: `--quiet` (capture only, no terminal stream),
`--fail-fast` (stop on first failure), `--no-summary` (skip summary
regeneration), `--config PATH` (alternate config file).

**Auto power-plan switching** (opt-in): `--auto-performance` captures
the current power plan / CPU governor at startup, switches to High
Performance (Windows: `SCHEME_MIN` GUID; Linux: writes `performance`
to `scaling_governor`), runs the sweep, and restores the original on
exit.

```
python common\orchestrator.py --auto-performance
```

Backup is written to `bench_out/.power_state_backup` before the switch.
On graceful exit (success, failure, SIGINT, SIGTERM) the original state
is restored and the backup is deleted. If the process is killed via
SIGKILL or the machine loses power, the backup remains and the next
orchestrator invocation detects it and prompts for restore.

**Linux requires root** because writing `scaling_governor` needs
privileged access. Without root, the switch silently fails and the
normal preflight abort takes over — set the governor manually or re-run
with `sudo`.

**Warning**: if the orchestrator is killed forcibly during a sweep, the
system stays in High Performance until you restore it manually
(`powercfg /setactive SCHEME_BALANCED` on Windows, or run the
orchestrator again to trigger the stale-backup prompt).

Expected runtime: ~15-25 minutes for all 13 radixes on a consumer
Raptor Lake class chip with AVX2 only.
