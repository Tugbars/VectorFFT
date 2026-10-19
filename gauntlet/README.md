# The gauntlet

Measure VectorFFT on your machine: calibrate the cells you care about through
the library's own front door (its race banks the verdicts as wisdom), time them
against MKL or FFTW or on their own, and get a report. One directory, one
driver, the library's own bench.

```
python gauntlet/gauntlet.py run --group pow2                 # every power of two 2..2^23
python gauntlet/gauntlet.py run --cells 4096                 # one cell
python gauntlet/gauntlet.py run --cells 2..4096              # a range
python gauntlet/gauntlet.py run --group primes --max 16384   # the primes to a ceiling
python gauntlet/gauntlet.py run --group mixed --max 4000000  # 2^a 3^b 5^c lengths
python gauntlet/gauntlet.py run --group pow2 --threads 8     # the same cells at 8 threads
python gauntlet/gauntlet.py run --group pow2 --inplace       # in place
```

## What a run does

1. **calibrate** -- one create per cell through `vfft_create` on a scratch copy
   of the shipped wisdom, this CPU's folder of it (`gauntlet/results/<run>/store/`).
   A cell the copy covers replays its verdict; a miss races and banks there.
   `--calibrate` re-races every cell (`config.recalibrate = 1`), so a run
   measures the library as it is now and never a verdict from an older build.
   The calibrate log records, per cell, whether it was **raced**, **replayed**
   or **refused** (no engine serves it).
2. **bench** -- the verdict the calibrate stage banked, replayed: one process
   per cell, both engine orders, best-of-5 in two
   timing windows, cachebust and a cool-down between engines, the caller pinned
   with its SMT sibling held, a control cell every 100 cells so a long run can
   be checked for drift. With MKL the ratio column; without it ns and GFLOPS.
   Correctness is measured on every cell (roundtrip error in the csv; `verify`
   checks the forward transform against a long-double reference).
3. **report** -- `report.md` in the run directory: every cell with how it was
   served, then the tables by route, size and family.
4. **merge** -- with `--merge`, the run's verdicts into the library's shipped
   wisdom, this CPU's folder of it (`src/wisdom/<folder>/`), with backups: a row
   keyed the same replaces the shipped one, a new row is added, rows the run did
   not touch are kept. Without it the verdicts stay in the run's own store (the
   `merge` verb merges a finished run later).

`gflops` turns a run into a GFLOPS list, VectorFFT beside MKL, one line per
cell (`gflops.csv` in the run directory; 5 N log2 N per transform, the best of
the two engine orders for each engine):

```
python gauntlet/gauntlet.py gflops --name gauntlet_pow2
python gauntlet/gauntlet.py gflops --name gauntlet_pow2 --threads 8
```

`verify` runs every cell's forward transform against a long-double scalar DFT
of the same input and writes the precision record `verify.csv` in the run
directory (`library,N,l2_error,max_error,rt_error`; with MKL built in, MKL's
rows on the same input, and with `--cmp fftw` FFTW's). The plots in `src/tools/plots/` read the run
directories: `gen_gflops.py` the bench csvs, `gen_precision.py` the verify csvs.

```
python gauntlet/gauntlet.py verify --name tw_exact_verify_2026-09-24 --cells 2..4096
python src/tools/plots/gen_precision.py gauntlet/results/tw_exact_verify_2026-09-24/verify.csv --out src/tools/plots/vectorfft-precision.svg
```

A stopped run resumes where it left off (`run` again with the same `--name`):
the cells it already re-raced and benched are not repeated, so a new run of
the same cells takes a new name.
The driver keeps the machine awake for the duration on Windows.

## Contracts

The default cell is the gauntlet's contract: 1D complex-to-complex, one
transform, natural order, out of place, one thread. `--threads T`, `--inplace`
are the other contracts; each writes its own csv and report (`gauntlet_ip.csv`,
`gauntlet_mt8.csv`, ...), never mixed into one table. Every ratio in a report
is comparator time / our time, the worse of the two engine orders.

**2D** (since 2026-09-23): a cell is a shape `N1xN2` (N1 = the column length),
the contract 2D complex-to-complex, interleaved, natural order, out of place,
K = 1, one thread, against MKL DFTI 2D out of place; files carry `_2d`
(`gauntlet_2d.csv`, `report_2d.md`, `calibrate_2d.log`, `verify_2d.csv`), the
control cell is 64x64, GFLOPS = 5 N1 N2 log2(N1 N2). Shapes never mix with 1D
lengths in one run. The groups:

```
python gauntlet/gauntlet.py run --group 2d-small [--max 64]   # every shape up to 64 per axis (3,969 cells)
python gauntlet/gauntlet.py run --group 2d-odd                # the odd/prime column pool and its closers x {64,128,256,512}
python gauntlet/gauntlet.py run --group 2d-pow2 [--max 8192] # the pow2 grid: every 2^a x 2^b up to 8192 per axis, planes to 4M points (159 cells)
python gauntlet/gauntlet.py run --group 3d-pow2 [--max 8192] # the 3D pow2 grid: every 2^a x 2^b x 2^c up to 8192 per axis, volumes to 4M points (1288 cells); the 3D contract (2026-09-24) mirrors the 2D one: --3d on the probes, --3dilnat on the bench, gauntlet_3d.csv / report_3d.md, the wisdom2_3d.txt shard
python gauntlet/gauntlet.py run --group 2d-mixed [--max 512]  # 2^a 3^b 5^c lengths as squares and against 64
python gauntlet/gauntlet.py run --cells 47x64,23x256          # any shapes
```

The report's tables are by route (`chain` / `blu`), by column class of N1
(pow2, even, odd, prime) and by plane size.

## Building the tools

Either path builds the same binaries from the same sources.

- CMake, with the library: `cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=gcc`
  then `cmake --build build --target vfft-gauntlet`; binaries in
  `build/gauntlet/`; run the driver with `--bin-dir build/gauntlet`.
- The gcc harness: `python gauntlet/build.py --compile --mkl --vfft --src
  gauntlet/bench_1d_vs_mkl.c` (and `recal_1d_probe.c`, `k1_fwd_ref_probe.c`
  with `--vfft`); binaries land beside the sources, which is the driver's
  default `--bin-dir`.
- The KFR arm: `KFR_ROOT=<KFR release package> python gauntlet/build.py --compile
  --mkl --kfr --vfft --src gauntlet/bench_1d_vs_mkl.c`, then `python
  gauntlet/gauntlet.py run --cells 2..4096 --cmp kfr`. The package is KFR's own
  release zip (for Windows `kfr-windows-x86_64.zip`: `include/kfr/capi.h`,
  `bin/kfr_capi.dll`); the arm, `gauntlet/kfr_arm.c`, calls KFR's C API, so the
  bench stays one build with our compiler and KFR runs the binary its authors
  built. The build copies the DLL beside the bench. It covers the 1D c2c cell at
  one thread, and its results land in `gauntlet_kfr.csv` with the same columns
  (the comparator column is KFR there). KFR is GPLv2 or commercial and is never shipped with this tree:
  the user supplies their own checkout, exactly as with MKL and FFTW.

- The FFTW arm: `--cmp fftw` on the driver (`python gauntlet/gauntlet.py run
  --cells 2..4096 --cmp fftw`), no build flag. The bench binds FFTW at run time
  by absolute path (`ref_fftw.h`: `$VFFT_FFTW_DLL`, else the FFTW in
  [`fftw/`](fftw/README.md) on Windows, else vcpkg's `fftw3.dll`) and refuses
  MKL's FFTW wrapper layer. The FFTW in `fftw/` is 3.3.10 with SSE2 to AVX-512
  and its threads API, so on Windows nothing needs installing. It serves the 1D, 2D and 3D c2c cells
  and the real cells (1D, 2D, 3D; in place too), out of place, FFTW_MEASURE, at
  one thread or `--threads T`; a threaded run needs an FFTW built with its
  threads API (`--enable-threads --with-combined-threads`, CMake
  `ENABLE_THREADS` + `WITH_COMBINED_THREADS`) -- the one in `fftw/` has it, the
  vcpkg DLL has none, and with such a DLL the bench exits 2 rather than time a
  serial FFTW against a threaded plan. FFTW's wisdom for the run lives in the run
  directory (`fftw.wis`), so one MEASURE verdict per problem serves every
  process of the run; results land in `gauntlet_fftw.csv` with the same
  columns (the comparator column is FFTW there), and `verify --cmp fftw`
  writes FFTW's rows into `verify_fftw.csv`.

MKL is optional at build time; FFTW is bound at run time. Without MKL the bench
reports absolute numbers and correctness (or FFTW's numbers under `--cmp fftw`).
With a comparator, the report says which library answered.

## Files

- `gauntlet.py` -- the driver (verbs: run, calibrate, bench, report, merge, cells, verify)
- `gauntlet_report.py` -- the report
- `bench_1d_vs_mkl.c` -- the canonical bench, every mode (K=1 interleaved, split layout, 2D, 3D, real, batches); `--2dilnat` is the 2D gauntlet cell
- `bench_1d_vs_fftw.c` -- a standalone FFTW bench; the gauntlet's FFTW comparator is the bench's `--cmp fftw` arm
- `recal_1d_probe.c` -- the calibrator (one front-door create; recalibrate re-races; `--2d N1 N2` for a shape)
- `k1_fwd_ref_probe.c` -- the forward reference check and the precision record (`verify`)
- `build.py` -- the gcc build harness (`src/tools/gates/run_gates.py` builds the gates with it too)
- `results/<run>/` -- cells.txt, store/, calibrate.log, gauntlet.csv, control.csv, verify.csv, gflops.csv, run.log, report.md
- `kfr_arm.c/.h`, `ref_fftw.h`, `bench_scope.h` -- the KFR arm, the run-time FFTW binding, the gauntlet's switches (`VFFT_BENCH_GUARD`, `VFFT_PCORE_MASK`) onto the library's measurement scope (`vfft.h`: the pin, the SMT-sibling guard, the priority, the machine's measurement lock)

The library's gates (checks that are not benches: kernels against a naive DFT, threaded
output bitwise against the serial plan, a door's pick against its incumbent) live in
`src/tools/gates/` and run through `python src/tools/gates/run_gates.py --only <gate>`,
which builds them with this folder's `build.py`.

Only the runs a document under `docs/performance/` or `docs/roadmap/` names stay under
`results/`; a run that fed no entry there is deleted once its number is recorded.
