# Benchmark scripts

Two scripts that run the benchmarks of the repository's front page (`README.md`, "Benchmark
Results") on your own machine, against FFTW or against Intel MKL, and draw the same five
pictures from your numbers:

| Picture | What is measured | File in the run directory |
|---|---|---|
| 1D throughput | the 1D complex transform at every length N from 2 to 4,096 (4,095 lengths), GFLOPS of both libraries | `vectorfft-<cmp>-gflops.svg` |
| 2D speedup matrix | the 2D complex transform on every power-of-two plane up to 4M points (159 planes) | `vectorfft-<cmp>-2d-pow2.svg`, `.png` |
| 2D threaded | the same 159 planes at T threads, the comparator at T threads too, beside the one-thread result; T = the physical cores of your machine's fastest core class | `vectorfft-<cmp>-2d-mt-forest.svg`, `.png` |
| 3D speedup matrices | the 3D complex transform on every power-of-two volume up to 4M points (1,288 volumes) | `vectorfft-<cmp>-3d-pow2.svg`, `.png` |
| Accuracy | the forward 1D transform at every N from 2 to 4,096 against a long-double reference, both libraries | `vectorfft-<cmp>-precision.svg` |

`<cmp>` is `fftw` or `mkl`. The contract is the front page's: complex-to-complex, double
precision, interleaved, natural order, out of place, one transform per call, one thread
(T threads for the third picture). The scripts drive the repository's own test tool, the
gauntlet (`gauntlet/`, described in `gauntlet/README.md`), with the commands the front page
names and `--cmp` added; they change none of its defaults. Everything they produce lands
under `gauntlet/results/`. Nothing is written into the library's shipped wisdom
(`src/wisdom/`) unless you run the gauntlet's `merge` verb yourself.

## What you need

| You need | Notes |
|---|---|
| an x86-64 CPU with AVX2 and FMA | and AVX-512 for an AVX-512 build (`--isa avx512`, below) |
| GCC or Clang | on Windows the MinGW-w64 build of GCC: the WinLibs UCRT build, or MSYS2's UCRT64 environment (`pacman -S mingw-w64-ucrt-x86_64-gcc`); Microsoft's compilers (`cl`, `clang-cl`) are not supported. The build looks in the usual MinGW-w64 and MSYS2 folders; `--cc` names another compiler |
| Python 3 | with `matplotlib` for the pictures (`pip install matplotlib`) |
| FFTW 3 as a shared library | for `--cmp fftw`: a `fftw3.dll` or `libfftw3.so`; which one, below |
| Intel oneMKL | optional, for `--cmp mkl`: an Intel oneAPI installation, found through `MKLROOT` or the default install path (`C:\Program Files (x86)\Intel\oneAPI\mkl\latest` on Windows) |
| Git | to get the sources: `git clone https://github.com/Tugbars/VectorFFT.git` |

CMake is not needed for this path: the tools are built by `gauntlet/build.py` with the
compiler alone. (The CMake target `vfft-gauntlet` builds the same binaries, see
`gauntlet/README.md`; the scripts here expect the ones `build_tools.py` builds.) The shell
scripts below run from bash (Git Bash or MSYS2 on Windows, bash on Linux) and from any
directory; the Python scripts they call also run from PowerShell, from the repository's
root.

## Which FFTW

The bench loads the FFTW shared library at run time, from the path you name: the FFTW you
measure is that file. When it binds, the bench reads FFTW's own version string, which names
the SIMD code paths the library was built with, for example
`fftw-3.3.10-sse2-avx-avx2-avx512`; `build_tools.py` prints the same string from the file,
so you know what you compare against before anything runs. FFTW's plans are made with
`FFTW_MEASURE`, and each run keeps its own FFTW wisdom file (`fftw.wis` in the run
directory), so a problem is planned once per run.

- A packaged FFTW serves the one-thread pictures. vcpkg's FFTW on Windows
  (`C:\vcpkg\installed\x64-windows\bin\fftw3.dll`, `fftw-3.3.10-avx-avx2-avx2_128`) has no
  threads API and no AVX-512 code paths. Linux packages often put the threads API in a
  second library (`libfftw3_threads.so`), which the bench cannot use: it needs the threads
  API in the one library it loads.
- The eight-thread picture needs an FFTW with the threads API in the same library (FFTW's
  "combined threads" build); without it the bench stops with a message, and `run_suite.py`
  refuses that set before anything runs. A comparison on AVX-512 wants an FFTW built with
  its AVX-512 code paths. For either, build FFTW 3.3.10 yourself from the source at
  fftw.org, with configure (MSYS2, Linux):

  ```sh
  ./configure --enable-shared --enable-sse2 --enable-avx --enable-avx2 --enable-avx512 --enable-threads --with-combined-threads
  make -j
  ```

  or with CMake:

  ```sh
  cmake -S fftw-3.3.10 -B fftw-build -DBUILD_SHARED_LIBS=ON -DENABLE_SSE2=ON -DENABLE_AVX=ON -DENABLE_AVX2=ON -DENABLE_AVX512=ON -DENABLE_THREADS=ON -DWITH_COMBINED_THREADS=ON
  cmake --build fftw-build --config Release
  ```

  The shared library the build leaves (a `.dll` on Windows, a `.so` on Linux) is the path
  you give `--fftw-dll`. Leave out `--enable-avx512` / `-DENABLE_AVX512=ON` on a CPU
  without AVX-512.

### If FFTW is not installed

With `fftw` as the comparator, `run_suite.py` looks for the library before anything runs,
in this order: `--fftw-dll`, the path `build_tools.py` recorded, the environment variable
`VFFT_FFTW_DLL`, then the usual place (vcpkg's `fftw3.dll` on Windows, the distribution's
`libfftw3.so.3` on Linux); it loads the first one found and reads its version. When none is
found, or the file does not load, it stops with exit code 2 and says how to point at one
and how to get one: on Windows `vcpkg install fftw3:x64-windows` (a serial AVX2 build), on
Linux the distribution's package (`apt install libfftw3-3`), and for the threaded picture or
AVX-512 the source build above. When the library has no threads API and the threaded set
was asked for, it refuses that set up front, the same way. `build_tools.py` makes the same
check of the library named with `--fftw-dll`. With `mkl` none of this runs.

## The two commands

### 1. Build the tools

```sh
./benchmark_scripts/build.sh --fftw-dll <path to the FFTW library>
```

`build.sh` calls `python benchmark_scripts/build_tools.py` with the same arguments
(`python3` when that is a working Python, else `python`):

| Option | Meaning |
|---|---|
| `--isa avx2` or `--isa avx512` | which of the library's two codelet trees to compile (default `avx2`; see "AVX-512" below) |
| `--mkl` / `--no-mkl` | build the MKL arm in, or leave it out; by default it is in when MKL is found |
| `--fftw-dll PATH` | the FFTW shared library for `--cmp fftw` runs, recorded for `run_suite.py` |
| `--cc NAME` | the compiler (default `gcc`) |

It builds the gauntlet's three tools (the bench `bench_1d_vs_mkl`, the calibrator
`recal_1d_probe` and the verify probe `k1_fwd_ref_probe`) through `gauntlet/build.py`, one
after the other, and prints each build's result line. The first build compiles the
library's codelets (about 1,600 files) once; that takes a few minutes, and later builds
reuse them until a codelet or the ISA changes. The binaries land in `gauntlet/`. What was
chosen (the ISA, whether MKL is in, the FFTW path and its version string, the compiler) is
recorded in `benchmark_scripts/.config.json`, which `run_suite.py` reads; the file is per
machine and not tracked by git. Build again with the other `--isa` to switch trees: the
record changes with it.

### 2. Run the benchmarks

One script per picture, and one for all five:

```sh
./benchmark_scripts/run_all.sh fftw --yes          # all five pictures, one after the other
./benchmark_scripts/run_1d.sh fftw --yes           # the 1D picture
./benchmark_scripts/run_accuracy.sh fftw           # the accuracy picture (after run_1d.sh, or on its own)
./benchmark_scripts/run_2d.sh fftw --yes           # the 2D picture
./benchmark_scripts/run_2d_mt.sh fftw --yes        # the 2D picture threaded (needs the run of run_2d.sh)
./benchmark_scripts/run_3d.sh fftw --yes           # the 3D picture
./benchmark_scripts/run_2d.sh mkl --yes            # the same against MKL
```

The threaded picture runs at as many threads as your machine has physical cores in its
fastest core class, read from the operating system and printed when the script starts: the
front page's own picture used 8 threads on an Intel Core i9-14900KF with 8 P-cores; on a
six-core Zen 4 the script uses 6. `--threads N` sets another count.

The first argument is the comparator, `fftw` (the default when left out) or `mkl`; every
further argument goes to `python benchmark_scripts/run_suite.py`, the script they all call
(`run_all.sh fftw --yes` is `run_suite.py --cmp fftw --yes`, `run_2d.sh mkl` is
`run_suite.py --cmp mkl --sets 2d`). Its options:

| Option | Meaning |
|---|---|
| `--cmp fftw` or `--cmp mkl` | the comparator (required; the shell scripts' first argument) |
| `--sets 1d,accuracy,2d,2d-mt,3d` | which sets to run (default all five; each `run_*.sh` names its own); they run in this order whatever order you list them |
| `--threads N` | the thread count of the `2d-mt` set (default: the physical cores of the fastest core class) |
| `--name-prefix NAME` | the run directories' prefix (default `<cmp>_<isa>_<date>`, e.g. `fftw_avx512_2026-10-19`) |
| `--fftw-dll PATH` | the FFTW library, when it is not the one `build_tools.py` recorded |
| `--yes` | answer the gauntlet's duration prompt in advance (a set estimated at more than 30 minutes asks) |
| `--plots-only` | draw the pictures and write the summaries from runs already finished; nothing runs |

Each set is one gauntlet command, then the plot script of that picture, drawn into the run
directory, then the summary. The gauntlet's output streams through as it runs; the first
failure stops the suite with the failing command printed, and the finished sets stay. A
stopped gauntlet run resumes where it left off when the same command runs again
(`gauntlet/README.md`), so running the same script again with the same `--name-prefix`
continues it; the prefix defaults to the day's date, so a run continued on another day
needs the prefix named. The `2d-mt` set draws the threaded run beside the one-thread `2d`
run, so it needs the `2d` run of the same prefix: `run_2d.sh` first, or `run_all.sh`.

## What lands where

One run directory per set under `gauntlet/results/`, the `1d` and `accuracy` sets sharing
one (the accuracy check reuses the 1D run's wisdom):

```
gauntlet/results/fftw_avx512_2026-10-19_1d/
    cells.txt                   the cells of the run
    store/                      the run's own copy of the wisdom: what the planner replayed, raced and banked for these cells
    calibrate_fftw.log          per cell: raced, replayed or refused, and the route
    gauntlet_fftw.csv           the measurement, one row per cell and engine order (the columns keep the names mkl_ns and ratio_vs_mkl whatever the comparator)
    control_fftw.csv            the control cell, re-timed every 100 cells, so drift shows
    report_fftw.md              the gauntlet's report: every cell, then the tables by route, size and family
    fftw.wis                    FFTW's wisdom for the run
    verify_fftw.csv             the accuracy record, one row per cell and library
    vectorfft-fftw-gflops.svg   the pictures
    vectorfft-fftw-precision.svg
    SUMMARY.md                  the machine, the comparator and its version string, the commands, the pictures, the report
gauntlet/results/fftw_avx512_2026-10-19_2d/        gauntlet_2d_fftw.csv, report_2d_fftw.md, vectorfft-fftw-2d-pow2.svg and .png, SUMMARY.md
gauntlet/results/fftw_avx512_2026-10-19_2d_mt6/    gauntlet_2d_mt6_fftw.csv, report_2d_mt6_fftw.md, vectorfft-fftw-2d-mt-forest.svg and .png, SUMMARY.md   (mt<T>: the thread count)
gauntlet/results/fftw_avx512_2026-10-19_3d/        gauntlet_3d_fftw.csv, report_3d_fftw.md, vectorfft-fftw-3d-pow2.svg and .png, SUMMARY.md
```

With `--cmp mkl` the files carry no `_fftw`: `gauntlet.csv`, `report.md`, `gauntlet_2d.csv`
and so on. The front page's own figures in `src/tools/plots/` are never touched. The run
directories are yours to keep, send or delete; the repository tracks the csv and reports of
the runs its documents cite, and yours only if you add them (their wisdom and logs are
ignored by git either way).

## How long it takes

Before each set `run_suite.py` prints the gauntlet's own estimate for the set's cells. The
estimate is the gauntlet's model from an Intel Core i9-14900KF and assumes every cell's plan
is replayed from the shipped wisdom. By that model:

| Set | Cells | Estimate |
|---|---|---|
| `1d` | 4,095 lengths | about 2.5 hours (calibrate 55 min, bench 100 min) |
| `accuracy` | 4,095 lengths | a few minutes (no estimate is printed) |
| `2d` | 159 planes | about 10 minutes |
| `2d-mt` | 159 planes | about 10 minutes |
| `3d` | 1,288 volumes | about 2.5 hours (calibrate 60 min, bench 90 min) |

The 1D sweep and the 3D grid take hours each; the whole suite is most of a day. On a CPU
the shipped wisdom has not seen, a cell is raced at create instead of replayed and
calibration takes longer than the estimate; the run's calibrate log says per cell which
happened. Leave the machine alone while a set runs: the bench pins itself to one core and
re-times a control cell every 100 cells so that drift shows in `control*.csv`, but it cannot
undo a busy machine.

## AVX-512

The library has two codelet trees, one per instruction set, and a build compiles one of
them (`--isa`). The interleaved library's AVX-512 tree is incomplete today: 716 kernels
against 982 for AVX2. Absent on AVX-512 are every real-transform engine (the real mono, real
pair, real flat with its Hermitian stage, and real rows engines; `codelets/zil/avx512/` under
`src/dag-fft-compiler/` has no `real/` folder), the tangent pair (`pair2p/tangent`), the
half column kernels (`shared/col/half`), 6 of the 18 blocked pair kernels, 6 of the 18
blocked column kernels and 4 of the 39 ZTT kernels (the radix-4 `t0tp` and `tld` forms,
which is why N = 16 and 32 have no ZTT route on AVX-512). The library never falls back from
AVX-512 to AVX2 for a kernel it lacks: such a kernel is simply absent, and the planner races
the routes that exist or refuses the cell. So on an AVX-512 build the five pictures here,
all complex-to-complex, are meaningful, while a real-transform cell may be refused. The
release will ship both ISAs.

To measure the AVX-512 tree, build the tools with `--isa avx512` on a CPU that has AVX-512,
against an FFTW built with its AVX-512 code paths (above); `build_tools.py` prints what the
FFTW file was built with, and `SUMMARY.md` records both.

On an AMD part such as Zen 4, the library reads the topology from the operating system
(physical cores, SMT siblings, core classes) as it does on Intel: every core counts as a
P-core, the thread pool pins one worker per physical core, and the measurement pins the
caller to one core and holds its SMT sibling the same way as on Intel.

## Files

- `build.sh` -- builds the tools: calls `build_tools.py` with its arguments
- `run_all.sh`, `run_1d.sh`, `run_accuracy.sh`, `run_2d.sh`, `run_2d_mt.sh`, `run_3d.sh` -- the benchmarks, all five or one picture each: the comparator first (`fftw`, the default, or `mkl`), then `run_suite.py`'s options
- `build_tools.py` -- builds the three tools through `gauntlet/build.py` and records `.config.json`
- `run_suite.py` -- runs the sets through `gauntlet/gauntlet.py`, draws the pictures with `src/tools/plots/gen_gflops.py`, `gen_2d_pow2.py`, `gen_forest_2d_mt.py`, `gen_3d_pow2.py` and `gen_precision.py`, and writes `SUMMARY.md`
- `.config.json` -- the record of the last build (per machine, not tracked)
