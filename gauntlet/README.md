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
   of the shipped wisdom (`gauntlet/results/<run>/store/`). A cell the shipped
   wisdom covers is replayed; a cell it does not is raced and banked there.
   `--calibrate` re-races every cell (recalibrate), which is how you calibrate
   the library for a different CPU. The calibrate log records, per cell, whether
   it was **raced**, **replayed** or **refused** (no engine serves it).
2. **bench** -- one process per cell, both engine orders, best-of-5 in two
   timing windows, cachebust and a cool-down between engines, the caller pinned
   with its SMT sibling held, a control cell every 100 cells so a long run can
   be checked for drift. With MKL the ratio column; without it ns and GFLOPS.
   Correctness is measured on every cell (roundtrip error in the csv; `verify`
   checks the forward transform against a long-double reference).
3. **report** -- `report.md` in the run directory: every cell with how it was
   served, then the tables by route, size and family.

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
rows on the same input). The plots in `src/tools/plots/` read the run
directories: `gen_gflops.py` the bench csvs, `gen_precision.py` the verify csvs.

```
python gauntlet/gauntlet.py verify --name gauntlet_2026-09-20 --cells 2..2048
python src/tools/plots/gen_precision.py gauntlet/results/gauntlet_2026-09-20/verify.csv --out src/tools/plots/vectorfft-precision.svg
```

`--merge` (or the `merge` verb) copies the run's verdicts back into the
library's shipped wisdom (`src/dag-fft-compiler/generator/generated/`), with
backups, so a calibration done on your host is kept.

A stopped run resumes where it left off (`run` again with the same `--name`).
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
python gauntlet/gauntlet.py run --group 2d-pow2               # squares 8..1024 and the rectangle ladder to 32768x64
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

MKL and FFTW are optional at build time. Without MKL the bench reports absolute
numbers and correctness. With both, the report says which library answered.

## Files

- `gauntlet.py` -- the driver (verbs: run, calibrate, bench, report, merge, cells, verify)
- `gauntlet_report.py` -- the report
- `bench_1d_vs_mkl.c` -- the canonical bench, every mode (K=1 interleaved, split layout, 2D, 3D, real, batches); `--2dilnat` is the 2D gauntlet cell
- `bench_1d_vs_fftw.c` -- the FFTW comparator
- `recal_1d_probe.c` -- the calibrator (one front-door create; recalibrate re-races; `--2d N1 N2` for a shape)
- `k1_fwd_ref_probe.c` -- the forward reference check and the precision record (`verify`)
- `build.py` -- the gcc build harness (the gauntlet's copy)
- `results/<run>/` -- cells.txt, store/, calibrate.log, gauntlet.csv, control.csv, verify.csv, gflops.csv, run.log, report.md
