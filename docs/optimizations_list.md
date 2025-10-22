1. Architectural Optimizations
True End-to-End SoA (The Big Win!)

Convert AoS→SoA ONCE at input, compute entirely in SoA, convert back ONCE at output
Eliminates 2 shuffles per stage per butterfly → saves 90% of shuffle operations
For 1024-pt FFT: 20 shuffles → 2 shuffles (10× reduction)

Out-of-Place Execution

Prevents read-after-write hazards in parallel execution
Enables ping-pong buffering between stages
Safer for streaming stores

2. SIMD Optimizations
Multi-Architecture SIMD Support

AVX-512: 8 doubles per vector (processes 4 complex values at once)
AVX2: 4 doubles per vector (processes 2-4 complex values)
SSE2: 2 doubles per vector (baseline x86-64)
Scalar fallback for cleanup/portability

FMA Instructions

fmsub and fmadd for complex multiply
4 FMA operations per complex multiply (optimal)
Reduces latency and instruction count

Architecture-Specific Conversion

AVX-512: Uses permutexvar_pd with valid indices (0-7)
AVX2: Uses unpacklo/hi_pd + permute4x64_epi64
Proper lane crossing handling

3. Memory Optimizations
Non-Temporal (Streaming) Stores

Enabled when write footprint > 70% of LLC
Minimum threshold: half ≥ 4096
Bypass cache for large transforms to prevent cache pollution
Runtime alignment checking with graceful fallback
Environment variable override (FFT_NT=0/1)

Cache-Aware Design

Chunk size: 64 complex values (8× cache lines)
Reduces false sharing in parallel execution
DOUBLES_PER_CACHE_LINE = 8 (64 bytes / 8 bytes)

Memory Alignment

Runtime alignment verification
Required alignment: 64B (AVX-512), 32B (AVX2), 16B (SSE2)
Graceful degradation if misaligned

Software Prefetching

Configurable prefetch distance: 24 elements default
Architecture-tuned: 16-32 elements ahead
Can be disabled if not beneficial

4. Parallelization
OpenMP Multi-Threading

Adaptive parallel thresholds based on SIMD width:

AVX-512: 2048
AVX2: 4096
SSE2: 8192
Scalar: 16384


Load balancing: parallelize the larger range
Cache-line-aligned chunks (PARALLEL_CHUNK_SIZE)

Memory Fence Optimization

_mm_sfence() only where needed (after streaming stores)
Per-thread fences in parallel path
No fence in sequential path (naturally ordered)

5. Special Case Optimizations
k=0 Optimization (W[0] = 1)

No twiddle multiply needed
Simple addition/subtraction: y[0] = x[0] ± x[half]
Scalar-only (single butterfly)

k=N/4 Optimization (W[N/4] = -i)

Multiply by -i → swap and negate: (re, im) * -i = (im, -re)
No trig computation needed
Scalar-only (single butterfly)

7. Performance Engineering
Heuristic-Based Optimization Selection

Automatic NT store enable/disable based on LLC size
Runtime detection vs compile-time decisions
Conservative defaults with override capability

Minimal Overhead Conversions

Per-element cost: 0.5-2.0 cycles depending on SIMD
For 1024-pt FFT: conversion overhead < 1%

Split-Range Processing

Two ranges: [1, k_quarter) and (k_quarter, half)
Parallelize larger range for better load balance