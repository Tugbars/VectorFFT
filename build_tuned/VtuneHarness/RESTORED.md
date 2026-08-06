# VtuneHarness — restored 2026-08-07

Recovered from commit `d55393ac` ("bench cleanup"), which deleted
`build_tuned/dev/bench_vtune/` in its entirety. The deletion was
unintentional (Tugbars). All seven files are restored **verbatim from
`d55393ac^`** — no edits — so this directory is the harness exactly as it
last worked.

| file | role |
|---|---|
| `bench_vtune.c` | 1D C2C profile bench, one **ITT task per cell** so VTune's task view side-by-sides VFFT and MKL on the same cell |
| `bench_vtune_cascade.c` | cascade-tier variant |
| `make_report.py` | composes `report.md` from the bench output + exported VTune CSVs — the durable artifact, readable without VTune installed |
| `run.bat` / `run_cascade.bat` | drivers: build, run under `vtune -collect <mode>`, export CSV, invoke `make_report.py` |
| `README.md` | original docs: collection modes, what to look for per cell class |
| `bench_output_hotspots.txt` | sample output from the last hotspots run (historical) |

## 🔴 It does NOT build against the current tree

Restored verbatim, which means it carries the **pre-`vfft_config_t` API**:

| harness uses | current API |
|---|---|
| `vfft_plan_c2c(N, K, flags)` | `vfft_config_t cfg; vfft_create(&cfg)` |
| `vfft_execute_fwd(p, re, im)` | `vfft_execute(h, VFFT_FORWARD, sre, sim, dre, dim)` |
| MKL `DFTI_REAL_REAL` + lane-major strides | that is the **split K-batch** contract |

The MKL side profiles split lane-major batches. For a K=1 **interleaved
in-place** question it must become `DFTI_COMPLEX` + `DFTI_INPLACE` +
`mkl_set_num_threads(1)`, and the vfft side `(z, NULL, z, NULL)`.

Its cell table is likewise the **old K-batch investigation** (131072×4,
8×256, Bluestein primes), not the current sub-2048 K=1 story.

**A ported copy targeting N = 256/512/1024/2048 at K=1 lives in**
`docs/research/mkl512_gap_campaign/vtune/` (gitignored). Port from there
rather than re-deriving; keep this directory as the clean reference.

## What is worth preserving here

- **ITT instrumentation.** Tasks named `VFFT_N{N}_K{K}_{CAT}` /
  `MKL_N{N}_K{K}_{CAT}` — filtering by task name isolates one cell, and the
  matching names put the two engines beside each other.
- **`auto_reps`** sizes each cell to ~2 s of work so sampling has enough
  population without a hand-tuned rep count per cell.
- **`make_report.py`** turns a VTune result dir into committable markdown.
- The README's "things to look for" section (top-down expectations per cell
  class, and the counters that mattered historically).

## Usage (once ported)

```cmd
build_tuned\VtuneHarness\run.bat                              :: build + run, no VTune
build_tuned\VtuneHarness\run.bat --collect uarch-exploration  :: port utilization + top-down
build_tuned\VtuneHarness\run.bat --collect hotspots           :: which codelet actually ran
```

VTune 2025.10 is installed at
`C:\Program Files (x86)\Intel\oneAPI\vtune\2025.10\bin64\vtune.exe`.

⚠ Verify the bench reproduces known ratios **before** trusting any VTune
attribution — a harness benching the wrong plan profiles the wrong code and
produces confident nonsense. Pin core 2 (mask `0x4`), HIGH priority.
