# Changelog

What changed from one release to the next, newest first. VectorFFT is in beta: a new
version can change how a function behaves, so compile your program again with each one.

## 0.9.1 beta (unreleased)

### Faster

- 2D real backward transforms (C2R) on several threads whose last extent is odd, such as
  512 x 2,039, 128 x 509 or 2,187 x 507: their rows now run on every thread, where only the
  column pass did. With the measuring planner and with `VFFT_FLAG_ESTIMATE`.
- With `VFFT_FLAG_ESTIMATE`, on several threads: 1D real transforms (R2C, C2R) from about
  524,288 points up whose half is not a power of two and has no prime factor above 47, such
  as 525,000, 1,058,400 or 8,403,500. The measuring planner already ran them this way.
- With `VFFT_FLAG_ESTIMATE`, on several threads: 2D real transforms (R2C, C2R) on planes
  larger than the CPU's L3 cache whose column length has a prime factor above 47, such as
  2,049 x 4,096 or 514 x 16,383, and tall planes larger than the L3 cache, such as
  8,192 x 1,024 or 8,193 x 1,018.
- Real transforms (R2C, C2R) whose first extent is computed with Bluestein's method (some
  lengths with a large prime factor, such as 2,063, 2,809 or 8,219): 2D and 3D transforms,
  ranks 4 to 7 and lane-major batches, on one thread or several, with the measuring planner
  and with `VFFT_FLAG_ESTIMATE`. That axis now runs as it does in complex transforms, through
  the 1D plan of its length on every column. 3D transforms with such a first extent now run on
  several threads, where they ran on one. With the measuring planner, the first create of such
  a transform times the new pass once and keeps the result in the wisdom.

### Fixed

- With `VFFT_FLAG_ESTIMATE`, `vfft_create()` refused 2D real forward transforms (R2C) whose
  column length is computed with Bluestein's method, such as 2,063 x 263 or 8,219 x 557: on
  one thread or several, out of place or in place. C2R and complex transforms were not
  affected.
- With `VFFT_FLAG_ESTIMATE`, `vfft_create()` refused some 2D real forward transforms (R2C) on
  one thread, out of place, whose last extent is an odd prime up to 2,048 and whose first
  extent is twice a length computed with Bluestein's method, such as 4,126 x 127,
  718 x 127 or 1,174 x 61.
- `vfft_destroy()` left part of a plan's memory allocated (a few megabytes a plan) for 2D
  complex transforms on several threads, one transform or a contiguous batch, whose first
  extent is computed with Bluestein's method, such as 2,063 x 64.

## 0.9.0 beta (2026-11-25)

### New

- Complex transforms (C2C) of rank 4 to 7: `rank` in `vfft_config_t` goes up to 7 and `n`
  holds seven extents. Interleaved layout, natural order, out of place or in place, on one
  thread or several, with the measuring planner or with `VFFT_FLAG_ESTIMATE`.
- Batches of complex transforms of rank 4 to 7 (`howmany` above 1): contiguous transforms by
  default, or lane-major with `VFFT_BATCH_LANE_MAJOR`.
- Real transforms (R2C, C2R) of rank 4 to 7, out of place, on one thread or several, with
  the measuring planner or with `VFFT_FLAG_ESTIMATE`.
- Batches of real transforms of rank 4 to 7 (`howmany` above 1), out of place: contiguous
  transforms by default, or lane-major with `VFFT_BATCH_LANE_MAJOR` at an even last extent
  of at least 4.
- DCT-II, DCT-III and DCT-IV (`VFFT_DCT2`, `VFFT_DCT3`, `VFFT_DCT4`) and the new DST-IV
  (`VFFT_DST4`): 1D, any length from 1 up, one transform a plan, out of place or in place, on
  one thread or several, with the measuring planner or with `VFFT_FLAG_ESTIMATE`. With them
  every DCT and DST of types I to IV is served.
- The split layout (`VFFT_LAYOUT_SPLIT`) in AVX-512 builds: complex transforms of rank 1, 2
  and 3 at the lengths the AVX2 build serves, one transform, contiguous batches and
  lane-major batches, out of place or in place, on one thread or several. Until now an
  AVX-512 build refused every split request.
- 1D complex transforms of lengths above 8,388,608 points with a prime factor above 47, such
  as 16,777,215 or the primes 8,388,617 and 16,777,213. 0.8.1 refused them.

### Faster

- 1D complex transforms of composite lengths with a prime factor above 47, such as
  2,097,151 or 4,194,305: above 262,144 points on one thread or several, and from 4,096
  points on several threads.
- 1D real transforms (R2C, C2R) on several threads whose length is twice such a length,
  from about 524,288 points up, and the DCT and DST types computed through them.
- 2D complex transforms on several threads: planes with composite or odd side lengths, such
  as 1,029 x 1,029 or 1,050 x 1,050, and tall narrow planes such as 875 x 15 or 2,625 x 6.
  Their speed no longer depends on where the threads' work buffers happen to sit in memory.
- 2D complex transforms on one thread with a short composite or odd column length and
  longer rows, such as 18 x 1,050, 42 x 1,050 or 35 x 1,029: out of place and in place,
  forward and backward, with the measuring planner and with `VFFT_FLAG_ESTIMATE`. 3D
  transforms made of such planes gain too.
- 2D complex backward transforms, most of all in place, on planes that fit in the CPU's
  caches.
- Side lengths with factors of 15, 21, 25 or 27, such as 75, 135, 525 or 675, in 2D and 3D
  transforms, and rows of 15, 21, 25 or 27 points.
- 2D complex transforms in natural order whose rows are a multiple of 512 bytes (64, 256 or
  1,024 points in double precision), such as 225 x 256 or 525 x 256, on one thread and on
  several.
- Planning 2D and 3D transforms with the measuring planner: the plan of a row length is
  measured once and kept in the wisdom, and every later plane or volume with rows of that
  length plans in a fraction of the time. Real transforms (R2C, C2R) share it where the
  last extent is even. Candidate plans that never win are no longer timed.

### Changed: adjust your program

- `VFFT_DST4` sits in `vfft_transform_t` after `VFFT_DST3`, so the number of `VFFT_DHT`
  (not served) went from 10 to 11.

### Fixed

- 1D complex transforms in scrambled order (`VFFT_ORDER_SCRAMBLED`) on two or more threads
  returned a wrong result at lengths of the form 2^k times an odd number, such as 10,000,
  50,000 or 100,000. AVX2 builds of 0.8.1 were affected; natural order, one thread, powers
  of two and AVX-512 builds were not.
- With `VFFT_FLAG_ESTIMATE`, `vfft_create()` refused a contiguous batch of 1D transforms on
  two or more threads at long lengths with a large prime factor: real transforms (R2C, C2R)
  from about 200,000 points up (200,003 and 262,146 are two such lengths), and complex
  transforms above 131,072 points on a processor whose L2 cache is larger than 2 MiB. One
  transform, one thread and the measuring planner were not affected.

### The DFTI interface

- The Trigonometric Transform routines (`mkl_trig_transforms.h`) serve MKL's staggered
  cosine, staggered2 cosine and staggered2 sine transforms, so all six of MKL's types are
  served in double precision.

## 0.8.1 beta (2026-11-19)

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
- Lane-major batches of 2D and 3D real transforms (R2C, C2R) of more than 16 transforms
  with short rows, such as 8 or 16 points.
- Planning a 2D transform for one thread with the measuring planner takes less time where
  the column length has only small prime factors.

### Changed: adjust your program

- Creating a plan for several threads with the measuring planner takes longer: the
  candidate plans are now timed on those threads.

### The DFTI interface

- MKL's Trigonometric Transform routines (`mkl_trig_transforms.h`, with `vfft_tt.c`
  compiled beside the library): the sine, cosine and staggered sine transforms, computed
  by VectorFFT's DST-I, DCT-I and DST-II/III with MKL's scales, and MKL's FFTW
  compatibility mode (`ipar[10] = 1`). The staggered cosine and staggered2 transforms and
  the single-precision routines are refused with MKL's status and a printed reason.

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
