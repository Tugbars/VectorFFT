# Baseline: VkFFT's CUDA backend

`vkfft_baseline` times VkFFT (1D, complex to complex, single precision, in place,
power-of-two lengths, batched) next to a plain in-place copy of the same buffer.
The copy reads every element once and writes it once: the least memory traffic a
one-pass transform can have. A transform that takes as long as the copy has no
time left to win.

## Build and run

```
cmake -S src/cuda -B <build dir> -G "Visual Studio 17 2022" -A x64 -DVKFFT_ROOT=<VkFFT checkout>
cmake --build <build dir> --config Release
```

The CUDA runtime DLLs (`<CUDA>/bin/x64`) must be on `PATH` to run it.

```
vkfft_baseline [--min-log2 K] [--max-log2 K]   lengths 2^K (default 3..16)
               [--total-log2 K]                 complex elements in the buffer (default 26 = 512 MB)
               [--rounds R] [--warmup W] [--queue-ms MS]
               [--csv FILE]                     one row per length
               [--lut]                          VkFFT reads twiddles from tables (its default here computes them)
               [--forward-only]                 lone forward launches only (for a profiler)
               [--dump-dir DIR]                 write VkFFT's generated source and binary, no timing
```

How a sample is taken is described at the top of
[vkfft_baseline.cpp](vkfft_baseline.cpp). Each run also checks VkFFT's forward
result against a double-precision host transform and the forward + inverse round
trip against the input.

## Results: RTX 5080, 2026-11-19

Folder [results/rtx5080_2026_11_19](results/rtx5080_2026_11_19). CUDA 13.1, driver
616.64, VkFFT 1.3.4, Windows 11. Medians of 31 rounds.

**512 MB buffer** (`vkfft_512MB.csv`, `.out`): time for one transform of the whole
buffer, and the faster of the two copy kernels.

| length | passes | VkFFT ms | copy ms | VkFFT / copy |
|-------:|:------:|---------:|--------:|-------------:|
| 8      | 1 | 1.450 | 1.424 | 1.02 |
| 64     | 1 | 1.476 | 1.369 | 1.08 |
| 256    | 1 | 1.441 | 1.434 | 1.00 |
| 1024   | 1 | 1.455 | 1.450 | 1.00 |
| 4096   | 1 | 1.431 | 1.434 | 1.00 |
| 8192   | 1 | 1.455 | 1.366 | 1.07 |
| 16384  | 2 | 2.908 | 1.387 | 2.10 |
| 65536  | 2 | 2.929 | 1.383 | 2.12 |

Every one-pass length (8 to 8192) moves the buffer at about 740 GB/s, read plus
write, the same as the copy. From 16384 on VkFFT needs two passes.

**Buffer size, length 4096** (`vkfft_batch_sweep.csv`, `.out`): time per queued
launch in microseconds. An empty kernel costs about 5.6 us per queued launch.

| buffer | transforms | VkFFT us | copy us | regime |
|-------:|-----------:|---------:|--------:|--------|
| 32 KB  | 1     | 7.4   | 5.7   | the launch itself |
| 2 MB   | 64    | 7.5   | 6.6   | the launch itself |
| 8 MB   | 256   | 10.0  | 9.1   | data stay in the GPU's L2 cache |
| 32 MB  | 1024  | 25.4  | 18.6  | data stay in the GPU's L2 cache |
| 64 MB  | 2048  | 129.8 | 145.2 | L2 cache exceeded |
| 128 MB | 4096  | 357.1 | 358.4 | device memory bandwidth |
| 512 MB | 16384 | 1364.9 | 1422.2 | device memory bandwidth |

**Profiles** (`ncu_*.out`, Nsight Compute 2025.4, one forward launch each). Nsight
Compute replays the launch on a cold cache, so its 32 MB figures are not the
cache-resident ones above.

| case | GPU time | device memory throughput | SM busy | registers / thread | shared memory / block | occupancy |
|------|---------:|-------------------------:|--------:|-------------------:|----------------------:|----------:|
| 4096, 512 MB | 1.28 ms | 86 % of peak | 14 % | 64 | 34.8 KB | 32 % |
| 1024, 512 MB | 1.76 ms | 85 % of peak | 10 % | 80 | 8.7 KB  | 21 % |
| 4096, one transform | 4.9 us | 2 % | 0.2 % | 62 | 34.8 KB | - |

In the 4096-point kernel 27 % of the single-precision arithmetic instructions are
fused multiply-adds; the other 73 % are separate.

**Every length, buffer in the L2 cache** (`vkfft_32MB.csv`, `vkfft_8MB.csv`, `.out`):
time per queued launch in microseconds on the 32 MB buffer, launch cost included
on both sides (5.7 us).

| length | passes | VkFFT us | copy us | VkFFT / copy |
|-------:|:------:|---------:|--------:|-------------:|
| 8 to 64 | 1 | 19.3 to 19.7 | 19.2 to 19.4 | 1.00 to 1.03 |
| 128    | 1 | 23.2 | 19.2 | 1.20 |
| 256    | 1 | 21.5 | 19.2 | 1.12 |
| 512    | 1 | 20.0 | 19.2 | 1.04 |
| 1024   | 1 | 21.5 | 19.4 | 1.11 |
| 2048   | 1 | 21.3 | 19.2 | 1.11 |
| 4096   | 1 | 24.8 | 19.3 | 1.28 |
| 8192   | 1 | 34.0 | 19.1 | 1.78 |
| 16384  | 2 | 71.7 | 19.8 | 3.63 |
| 65536  | 2 | 76.0 | 19.5 | 3.90 |

**Twiddles computed or read from tables** (`vkfft_32MB_twiddles_*.csv`,
`vkfft_512MB_twiddles_*.csv`): forward error against the double-precision host
transform (relative, root mean square), and time on the 32 MB buffer. On the
512 MB buffer the two take the same time.

| length | error, computed | error, tables | us, computed | us, tables |
|-------:|----------------:|--------------:|-------------:|-----------:|
| 64   | 1.6e-7 | 0.9e-7 | 19.5 | 22.1 |
| 1024 | 3.1e-7 | 1.3e-7 | 21.5 | 22.7 |
| 4096 | 4.2e-7 | 1.4e-7 | 25.7 | 25.7 |
| 8192 | 4.4e-7 | 1.5e-7 | 36.5 | 38.2 |
