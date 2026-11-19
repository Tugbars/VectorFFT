# Changelog

What changed from one release to the next, newest first. VectorFFT is in beta: a new
version can change how a function behaves, so compile your program again with each one.

## 0.8.1 beta (unreleased)

### Faster

- 1D transforms on several threads with the measuring planner: the method and its
  factorization are chosen by timing the candidates on those threads, not on one thread.
- 2D complex transforms on several threads with the measuring planner: the factorization of
  the columns, the method of the rows and the way the threads share the plane are chosen
  together, each timed on the whole transform on those threads. Large 1D transforms
  computed in four steps gain from it too.
- 1D lengths of the form 2^k times an odd number (such as 10,000, 20,480 or 120,000) on
  several threads.
- Lengths computed with Rader's or Bluestein's method (lengths with a large prime factor) on
  several threads: the inner transform is chosen for the threads, and a large inner
  transform now runs on the threads too. The same holds for real transforms (R2C, C2R)
  whose half length is such a length.
- 1D real transforms (R2C, C2R) whose length is not a power of two, on several threads: the
  complex transform inside them runs on the threads too.
- 2D transforms and lane-major batches whose column length is a prime (such as 1031, 4099
  or 4483), on several threads.
- Column lengths with several factors of 5 or 7 (such as 6,860, 12,005 or 16,807) in 2D
  transforms and lane-major batches, on one thread and on several.
- Lane-major batches on several threads, most of all batches of few transforms.
- Planning a 2D transform for one thread with the measuring planner takes less time where
  the column length has only small prime factors.

### Changed: adjust your program

- Creating a plan for several threads with the measuring planner takes longer: the
  candidate plans are now timed on those threads.

## 0.8.0 beta (2026-11-17)

### Faster

- 2D transforms on several threads with `VFFT_FLAG_ESTIMATE`: a plane that fits the CPU's
  L1 data cache runs on one thread where threads do not pay off, and is split across the
  threads where they do.
- Lengths computed with Bluestein's method (lengths with a large prime factor) with
  `VFFT_FLAG_ESTIMATE`: a faster inner transform, in 1D, in the rows of 2D transforms and
  in real transforms.
- 1D lengths above 262,144 with a large prime factor with `VFFT_FLAG_ESTIMATE`: planned the
  way the measuring planner plans them.
- On AVX-512 CPUs, in natural order: most 1D lengths from 2,160 to 252,000 whose
  power-of-two part is 16 or 32.
- Lane-major batches of 2D and 3D real transforms (R2C, C2R) with few transforms in the
  batch or short rows, and 1D real batches of two transforms.
- Lane-major batches with `VFFT_FLAG_ESTIMATE`: better plans, on AVX2 and AVX-512.
- Lane-major batches with the measuring planner: each candidate plan is timed the way the
  batch runs it, so the planner keeps the faster one.
- Lane-major complex batches of 64 transforms or more, and 2D and 3D column passes as wide,
  whose rows are a multiple of 512 bytes: faster, and no longer faster or slower depending
  on where the buffers happen to sit in memory.

### Changed: adjust your program

- `vfft_execute`, `vfft_execute_split` and `vfftf_execute` return an `int`: 0 when the
  transform ran, -1 when the call was refused (nothing was computed or written;
  `vfft_error()` says why). They returned nothing before.
- An out-of-place execute whose input and output buffers partly overlap is refused: it
  returns -1 and writes nothing. Before, 2D, real and batch plans gave wrong results. One
  buffer for both (`out == in`) is still served.
- `wisdom_dir = ""` is refused at create. Before, it was taken for `NULL` and the plans
  went into the library's own wisdom.

### Fixed

- A batch count whose size overflows (for example `howmany = SIZE_MAX`) is refused at
  create.
- Lane-major batches of more than 2^31 transforms plan and run; a contiguous batch of more
  than 2^31 planes ran only 2^31 of them.
- Running out of memory while planning is refused with that reason, instead of a crash or
  another reason.
- A SPLIT request in an AVX-512 build is refused with the real reason: the split layout is
  AVX2-only in this version.
- An `nthreads` outside the thread pool's range gets the same message with
  `VFFT_FLAG_ESTIMATE` as when measuring.
- The wisdom folder's first message says "writable" only when it is; a missing or
  read-only folder is reported once, at the first save.
- A memory leak while measuring some plans.
- A lane-major batch whose rows use Bluestein's method: the plan read back from wisdom is
  the one the first run measured.

### Benchmark programs

- The download: one file per test (`run_selected`, `run_pow2`, `run_selected_planned`,
  `run_pow2_fp32`), a `.bat` to double-click on Windows and a `.sh` on Linux. They replace
  the menu.
- From the sources on Windows: one double-click `.bat` per benchmark set in
  `benchmark_scripts/scripts/` (`build_and_run_1d.bat`, `_accuracy`, `_2d`, `_2d_mt`,
  `_3d`, `_all`). The shell scripts there are removed; `benchmark_scripts/README.md` gives
  the Python commands they ran.

### License

- Every source file of the library and of its kernel generator carries an SPDX header
  naming its license (MIT).

## 0.7.0 beta (2026-11-15)

First public release.
