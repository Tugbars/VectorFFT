# bench_vtune — VTune-instrumented 1D C2C profile bench

Profiles 12 hand-picked cells from the 207-cell MKL bench, spanning
the regime where VectorFFT is borderline (CLOSE: ratio < 1.30× over
MKL — investigate ILP / store-bound ceilings) through where it
dominates (DECISIVE: ratio > 4× — validate codelet behavior).

Each cell runs ~2 seconds of FFT work bracketed by **ITT API tasks**
so VTune attributes samples (retiring %, port pressure, machine clears,
DTLB misses, etc.) to a named region in the GUI's task view.

## Cells

| Category | N | K | Expected ratio | Why this cell |
|----------|--:|--:|---------------:|---------------|
| CLOSE | 131072 | 4 | 1.17× | pow2 K=4 ILP weakness — our worst pow2 cell |
| CLOSE | 32768 | 4 | 1.20× | pow2 K=4 |
| CLOSE | 8192 | 4 | 1.32× | pow2 K=4 borderline |
| CLOSE | 243 | 4 | 1.26× | 3^5 prime power, K=4 |
| MID | 1024 | 256 | 1.94× | pow2 baseline, well-tuned |
| MID | 2048 | 256 | 2.08× | pow2 mid |
| MID | 4096 | 256 | 1.80× | pow2 large |
| DECISIVE | 8 | 256 | 7.78× | small-N batch, MKL drowns |
| DECISIVE | 16 | 256 | 4.80× | small-N batch |
| DECISIVE | 60 | 32 | 5.15× | composite 12×5 |
| DECISIVE | 128 | 256 | 2.93× | radix-4×4×8 |
| DECISIVE | 243 | 256 | 2.69× | same N=243 as CLOSE, contrast K dim |

## Quick check (no VTune)

```cmd
build_tuned\dev\bench_vtune\run.bat
```

Just builds + runs the bench, prints per-cell wall time + GFLOP/s + ratio.
Use to confirm the build works and the cells produce expected numbers
before running under VTune.

## Profiling workflows

### Microarchitecture exploration (most useful for codelet analysis)

```cmd
build_tuned\dev\bench_vtune\run.bat --collect uarch-exploration
```

Captures port utilization, retiring % per task, frontend / backend bound
breakdown, machine clears, DTLB miss rate. Best for diagnosing the
CLOSE cells — figuring out *why* MKL is competitive at large pow2 K=4
(typically ILP-bound on the codelet's stage chain).

### Hotspots (which functions dominate per cell)

```cmd
build_tuned\dev\bench_vtune\run.bat --collect hotspots
```

Function-level CPU time. Use to validate the DECISIVE cells route
through the expected codelet variants (`radix*_t1_dit_*`, `radix*_t1s_*`)
and that twiddle / executor overhead is amortized.

### Threading

```cmd
build_tuned\dev\bench_vtune\run.bat --collect threading
```

Not very useful here — the bench runs single-threaded by design (T=1
makes codelet behavior visible without thread-pool obscuration).
Re-enable MT in `bench_vtune.c` (`vfft_set_num_threads(8)`) before
collecting threading data.

## Reading the results

After a `--collect` run, the script:
1. Captures the bench's per-cell timing output to `bench_output.txt`
2. Exports VTune reports as CSV (`summary.csv`, `hotspots.csv`,
   `topdown.csv` for uarch mode)
3. Composes a markdown report at `<result_dir>/report.md` combining
   bench timings + per-task microarchitecture breakdown + top hotspot
   functions per cell

The markdown report is the durable artifact — commit it to the repo or
share with collaborators without needing VTune installed to read it.

For richer interactive analysis open the result dir in the VTune GUI:

```cmd
"C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune-gui.exe" ^
    build_tuned\dev\bench_vtune\vt_uarch-exploration
```

Each task is named `VFFT_N{N}_K{K}_{CATEGORY}` or `MKL_N{N}_K{K}_{CATEGORY}`
so the task view side-by-sides VFFT and MKL on the same cell. Filter by
task name to focus on one cell at a time.

## Things to look for

**CLOSE cells (where MKL is competitive):**
- High **Backend Bound — Memory Bound — Store Bound** %: large pow2 K=4
  expects this (Cf. `docs/vtune_full_fft_n16384_k4.md` analyzing N=16K
  K=4 — that file showed 32.4% DTLB store bound).
- **Bad Speculation** elevated: cf. R=11 codelet's 4.4% machine clears
  (per `memory/MEMORY.md`).
- **Frontend Bound — DSB Bandwidth**: codelet tail too big for the µop
  cache; possible v1.1 codegen tweak.

**DECISIVE cells (validate codelets):**
- **Retiring** ≥ 60% expected on small-N: codelets fully fed, port
  utilization high. Cf. `docs/vtune_r4_codelet_k256.md` — R=4 hit 86%
  retiring.
- **Port 0/1 utilization** ≈ 90% on FMA-heavy radices.
- **No spills** — confirmed via L1 store rate. If you see high L1 store
  bandwidth on a DECISIVE cell, something regressed.

**MID cells:**
- Sanity check that MKL and VFFT take the same algorithmic path
  (mixed-radix CT, no Bluestein/Rader) and the only difference is
  codelet / variant quality.

## Build details

- Compiles with **MSVC** + AVX2 (the cl.exe path links cleanly against
  the ICX-built `vfft.lib` once Intel oneAPI's LIB path is on env).
- Links **MKL ILP64 sequential** for side-by-side. Add `--mkl` arg
  to enable both VFFT and MKL benches; run without to do VFFT only.
- Links **libittnotify.lib** from VTune SDK so the ITT markers compile
  out to no-ops if VTune isn't running (~1 cycle per task call) and
  attribute samples to the named regions when it is.
- **Power plan** is set to High Performance on entry, restored on exit.
  Run from elevated cmd if `powercfg /setactive` fails (admin needed).
- **Single-threaded** by default (`vfft_set_num_threads(1)`,
  `mkl_set_num_threads(1)`). VTune profiles are clearer at T=1; thread
  pool dispatch obscures codelet attribution.

## When to re-run

- After codelet generator changes (`gen_radix*.py`) — confirm the new
  codelets retire at expected rates on the DECISIVE cells.
- After cost-model changes — confirm the CLOSE cells haven't gotten
  worse (the cost model picks the factorization; if it regresses, the
  bench will surface it via increased Backend Bound %).
- Before tagging a release — sanity check no cell regressed in
  microarchitectural quality.
