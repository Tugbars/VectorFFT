# highspeedFFT

A high-performance mixed-radix FFT library optimized with AVX2 SIMD instructions, supporting both power-of‑N and arbitrary-length transforms (via Bluestein’s algorithm). This implementation emphasizes contiguous memory layouts, precomputed twiddle tables, and robust bounds checking.

## Features

* **Mixed‑radix DIT FFT** up to radices 2, 3, 4, 5, 7, 8; general radix beyond 8.
* **Bluestein’s algorithm** for arbitrary lengths, padded to a power of two.
* **AVX2 SIMD acceleration**:

  * Flattened real/imag arrays for contiguous loads/stores.
  * Unaligned `_mm256_loadu_pd`/\_`storeu_pd` for robustness.
  * Vectorized butterfly loops processing 4 complex samples at a time.
* **Precomputed tables**:

  * Twiddle lookup tables for common radices (2, 3, 4, 5, 7, 8, 11, 13).
  * Bluestein chirp sequences up to N=64.
* **Safety & correctness**:

  * Bounds checks on twiddle-array sizes.
  * Heap allocation with error handling and no leaks.

## Directory Structure

```text
include/           # Public headers (highspeedFFT.h)
src/               # FFT implementation (mixed_radix_dit_rec, bluestein_fft)
bench/             # Benchmark scripts
tests/             # Unit and integration tests
CMakeLists.txt     # Build configuration
README.md          # This document
```

## Building

This project uses CMake:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j    # produces libhighspeedfft.a and examples
```

## Usage

```c
#include "highspeedFFT.h"

// Initialize
fft_object fft = fft_init(N, +1);  // forward FFT
fft_data *in  = malloc(N*sizeof(*in));
fft_data *out = malloc(N*sizeof(*out));
// fill in...
fft_exec(fft, in, out);

// cleanup
free_fft(fft);
free(in); free(out);
```

## Performance Comparison: highspeedFFT vs FFTW

| Library      | Radices                | AVX2 Vectorization | Twiddle Layout       | Throughput (e.g. 1M points/s)        |
| ------------ | ---------------------- | ------------------ | -------------------- | ------------------------------------ |
| highspeedFFT | mixed‑radix, Bluestein | ✓                  | flat `out_re/out_im` | \~X× faster on small and mixed sizes |
| FFTW         | radix‑2                | ✓ (SSE/AVX)        | interleaved, gather  | baseline                             |

* **Radix flexibility**: FFTW primarily optimizes powers-of-2; highspeedFFT adds efficient radices 3, 5, 7, 8 and arbitrary lengths without sacrificing speed.
* **Memory layout**: contiguous real/imag arrays in highspeedFFT yield faster loads than FFTW’s gather-based approach for non‑power‑of‑2.
* **Bounds safety**: explicit checks on twiddle sizes prevent out-of-bounds errors.

Benchmark your own workloads under `bench/` to see detailed comparisons.


