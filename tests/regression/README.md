# Regression test

The test is called regression, and its cells can serve as one, but its goal is
to check two things across different platforms:

1. **The estimate's accuracy, per ISA.** How close the estimate's plans
   (`VFFT_FLAG_ESTIMATE`: formulas, nothing measured) come to the planner's on
   each platform, for each instruction set that platform runs.
2. **The efficiency of each ISA.** How the AVX2 build compares with the AVX-512
   build of the same cells on one machine, and whether that holds from one
   platform to the next.

Both are ratios taken on one platform: the estimate's time over the planner's,
and the AVX2 build's time over the AVX-512 build's. Platforms are compared
through those ratios, never through raw times.

## The cells

| File | Cells | Sizes |
|---|---|---|
| `octaves.txt` | 180 | 1D lengths, 8 to 2^24 points |
| `2d-octaves.txt` | 145 | 2D shapes `N1xN2` (N1 is the column length), planes of 2^7 to 2^23 points |
| `3d-octaves.txt` | 85 | 3D shapes `N1xN2xN3`, volumes of 2^9 to 2^23 points |

Every platform runs these same cells. They come from arithmetic alone: octave
edges, primes and factorizations. No platform's caches, planner or measurements
went into choosing them.

- **Classes:** a length is one of:
  - a power of two;
  - 2^a × odd, every prime factor at most 47;
  - odd, every prime factor at most 47;
  - a prime above 47;
  - a composite with a prime factor above 47.
- **1D:** each class has two cells per octave [2^k, 2^(k+1)), one at its
  beginning and one at its end. A power of two has one, 2^k.
- **2D and 3D:** the axes are those lengths.

`make_cells.py` states the rules in full and writes the three files. Each line
of a file names its cell's class, octave and factors. The machinery the cells
cover is drawn in
[`docs/architecture/machinery/`](../../docs/architecture/machinery/).

A cell's cache level differs from platform to platform. It is labelled when a
platform's results are read, from that platform's own L1d, L2 and L3, and is
never used to choose cells.

## Running it

The gauntlet runs the cells ([`gauntlet/README.md`](../../gauntlet/README.md)).
Each file runs twice into one run directory: first with the planner's plans,
then with the estimate's (`--est`, whose files carry `_est`).

```
python gauntlet/wisdom_folder.py     # names this CPU's wisdom folder, src/wisdom/<folder>
python gauntlet/gauntlet.py run --cells @tests/regression/octaves.txt --name regression_1d_avx2 --bin-dir build-avx2/gauntlet --store src/wisdom/<folder>
python gauntlet/gauntlet.py run --cells @tests/regression/octaves.txt --name regression_1d_avx2 --bin-dir build-avx2/gauntlet --est
python gauntlet/gauntlet.py run --cells @tests/regression/octaves.txt --name regression_1d_avx2 --bin-dir build-avx2/gauntlet --est --batch contiguous --k 4
python gauntlet/gauntlet.py run --cells @tests/regression/octaves.txt --name regression_1d_avx2 --bin-dir build-avx2/gauntlet --est --batch lane-major --k 4
python gauntlet/gauntlet.py run --cells @tests/regression/octaves.txt --name regression_1d_avx2 --bin-dir build-avx2/gauntlet --est --batch contiguous --k 8
python gauntlet/gauntlet.py run --cells @tests/regression/octaves.txt --name regression_1d_avx2 --bin-dir build-avx2/gauntlet --est --batch lane-major --k 8
```

- **The planner's run** calibrates first. A cell the store holds replays its
  verdict, and a missing cell races and banks into this CPU's store.
- **The estimate's run** reads and banks no wisdom.
- **The 2D and 3D files** take the same two runs.
- **Threads:** add `--threads T` for T = 2, 4 and 8, up to the platform's
  cores. Each T writes its own files (`_mt4` and so on) in the same run
  directory.
- **Batches:** each file also runs as a batch of 4 transforms and as a batch
  of 8, in both layouts (`--batch contiguous` and `--batch lane-major`, with
  `--k 4`, then `--k 8`), at one thread unless asked. A cell whose K
  transforms pass 2^24 points is not benched. That leaves 162 of the 1D cells,
  110 of the 2D and 66 of the 3D at K = 4, and 153, 110 and 51 at K = 8. For
  now the batches run with the estimate's plans only.
- **Duration:** the `cells` verb in place of `run` lists the cells and
  estimates a run's duration without running anything.

Each instruction set is its own build of the gauntlet's tools, with its own run
directory. A platform without AVX-512 runs the AVX2 build alone. Pass
`--bin-dir` every time: without it the driver takes the first build folder
that holds a bench, which need not be the fresh one.

```
cmake -S . -B build-avx2 -G Ninja -DCMAKE_C_COMPILER=gcc -DVFFT_ISA=avx2
cmake --build build-avx2 --target vfft-gauntlet
cmake -S . -B build-avx512 -G Ninja -DCMAKE_C_COMPILER=gcc -DVFFT_ISA=avx512
cmake --build build-avx512 --target vfft-gauntlet
```

## Reading the results

A cell's time is the better of its two engine orders (`vfft_ns` in the run's
csv).

- **The estimate's accuracy:** in one run directory, the cell's time in
  `gauntlet_est.csv` over its time in `gauntlet.csv`, and likewise
  `gauntlet_2d_mt4_est.csv` over `gauntlet_2d_mt4.csv`. 1.0 means the estimate
  found a plan as fast as the planner's. The `route` columns name the machinery
  each one picked.
- **The efficiency of each ISA:** the cell's time in the AVX2 run directory
  over its time in the AVX-512 run directory, on the same machine.
- **Batches:** their files (`gauntlet_k4_est.csv`, `gauntlet_2d_k8_lm_est.csv`
  and so on) are read the same way. Until the planner's batch runs exist, they
  give the ISA comparison only.
- **Across platforms:** those ratios side by side, by class and by each
  platform's own cache level.

Two runs compare only when their control cells agree (`control.csv` against
`control_est.csv`, and the two run directories' control files).
