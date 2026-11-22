# Rank 4 to 7: the tests

Checks and timing cells for complex transforms of rank 4 to 7 (`VFFT_C2C`, interleaved, natural order), one
transform or a batch. The transforms themselves are the rank-N tier,
[`src/core/il/rank3/fftnd_il.h`](../../src/core/il/rank3/fftnd_il.h); a contiguous batch is the queue over its plan
([`plane_queue.h`](../../src/core/il/rank2/plane_queue.h)), a lane-major batch its own door
([`lm_batch3d_il.h`](../../src/core/il/rank3/lm_batch3d_il.h)).

## What is here

| File | What it does |
|---|---|
| [`nd_probe.c`](nd_probe.c) | One shape through the public API: the forward output against a reference that shares nothing with the tier, and the round trip. |
| [`nd_sweep.py`](nd_sweep.py) | The correctness sweep: `nd_probe` over a fixed list of 88 shapes. |
| [`make_cells.py`](make_cells.py) | Writes the timing cells. |
| `4d-cells.txt` .. `7d-cells.txt` | The timing cells, one file a rank: 36, 21, 17 and 14 shapes. |

## The correctness sweep

Build the gauntlet target first (it builds `build/gauntlet/libvfft_core.a`), then the probe:

```
gcc -O2 -Iinclude tests/rank_n/nd_probe.c build/gauntlet/libvfft_core.a -lm -o build/nd_probe.exe
python tests/rank_n/nd_sweep.py
```

- **The reference.** A shape of up to 70,000 points is checked against the direct DFT along every axis, in long
  double. A larger one is checked against the library's own 1D plans applied line by line.
- **The runs.** Every shape runs with an estimate plan (`VFFT_FLAG_ESTIMATE`) out of place on one thread; a part of
  them also in place and at 4 and 8 threads.
- **The batches.** A shape of up to 300,000 points also runs as a batch at 1 and 8 threads: three contiguous
  transforms, or four lane-major ones. Every transform of the batch is checked against the reference.
- **The shapes.** Powers of two, composites, odd extents, a prime above 47 on each axis in turn, extents of 2 and
  3, and arrays up to 16.7 million points.
- **Measured plans.** `--store DIR` also plans a part of the list by measuring, batches among them, in the scratch
  wisdom folder `DIR`.
  The probe refuses a measured plan without `--store`, so a test never plans into the library's own store.
- **The result.** One line a failure, then a summary; the exit status is 1 when any run failed. A run fails when
  either error passes 1e-11.

`nd_probe` alone prints one line a shape: the two errors, the forward's best time, and `sum=`, a hash of the forward
output's bits. Two builds that print the same sum for a shape ran the same arithmetic.

## The timing cells

Four classes at each rank: every extent a power of two; extents 2^a × odd; odd extents; a prime above 47 on one
axis. Each class runs from a few KiB to the largest array of the rank under 2^24 points, as cubes and as shapes
with one long or one short axis. The list comes from arithmetic alone, so every machine times the same shapes.

The gauntlet times one rank a run, with estimate plans:

```
python gauntlet/gauntlet.py run --cells @tests/rank_n/4d-cells.txt --name <run> --bin-dir build/gauntlet --est
```

Add `--threads T` for a threaded run and `--cmp fftw` for FFTW in MKL's place. The bench's cell is `--ndnat`
(`gauntlet/bench_1d_vs_mkl.c`): the 3D cell's protocol at the shape's rank.
