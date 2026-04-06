# Why VectorFFT Outscales MKL at Multithreading

## Results Summary

8 threads, i9-14900KF (P-cores pinned), both libraries at 8 threads:

| N | K | VectorFFT (ns) | MKL (ns) | vs MKL |
|---|---|---|---|---|
| 200 | 4096 | 278,094 | 3,343,052 | **12.02x** |
| 875 | 256 | 70,181 | 923,624 | **13.16x** |
| 1000 | 256 | 75,144 | 945,052 | **12.58x** |
| 1000 | 2048 | 1,004,709 | 10,451,813 | **10.40x** |
| 5000 | 2048 | 13,400,892 | 73,526,146 | **5.49x** |
| 10000 | 256 | 1,408,537 | 13,833,922 | **9.82x** |

MKL is actually **slower** with 8 threads than single-threaded at K=256 (945K vs 809K ns for N=1000). VectorFFT gets 4.25x speedup at the same size.

## Why MKL's Threading Hurts

### 1. Wrong parallelization axis

MKL parallelizes **within a single N-point transform** — it splits the Cooley-Tukey decomposition across threads. With `DFTI_NUMBER_OF_TRANSFORMS = K`, MKL loops over K internally and parallelizes the N-point stages. This means:

- All threads share the same N-point data
- Cache coherency traffic between cores (MESI protocol bouncing)
- Inter-stage barriers are mandatory (stage s+1 depends on s)

VectorFFT parallelizes **across the K batch dimension**. Each thread owns a disjoint slice of memory. Zero sharing, zero coherency traffic.

### 2. Split-complex layout advantage

VectorFFT stores data as `re[n * K + k]` — K contiguous doubles per butterfly leg. Slicing by K gives each thread a contiguous, cache-line-aligned memory region. No stride adjustments needed.

MKL's internal representation requires reorganizing the batched split-complex data to fit its within-transform parallelism model, adding overhead.

### 3. Thread dispatch overhead

MKL uses OpenMP (`libiomp5`) for threading. OpenMP dispatch involves:
- Runtime thread selection
- Work-sharing construct setup
- Implicit barriers at parallel region boundaries

VectorFFT uses a **spin-wait thread pool**:
- Workers spin on a volatile flag (~10ns wake latency)
- Dispatch is a single memory write
- No OS primitives in the hot path

For sub-millisecond FFT calls (our typical range), MKL's dispatch overhead is a significant fraction of total time.

### 4. Strategy selection

VectorFFT automatically selects the best strategy per call:

| Condition | Strategy | Overhead |
|---|---|---|
| K/T >= 256 | K-split | Zero barriers, zero copies |
| K/T < 256 | Group-parallel | Spin-barrier between stages (~100ns each) |
| K < 4 | Single-threaded | No dispatch |

MKL uses one strategy (within-transform) regardless of K.

## VectorFFT Threading Scaling

| N | K | T=1 | T=8 | Speedup | Strategy |
|---|---|---|---|---|---|
| 1000 | 256 | 478,340 | 112,601 | **4.25x** | group-parallel |
| 1000 | 512 | 1,075,126 | 226,927 | **4.74x** | group-parallel |
| 200 | 4096 | 1,723,257 | 287,113 | **6.00x** | K-split |
| 1000 | 1024 | 2,528,186 | 681,190 | **3.71x** | K-split |

K-split scaling is limited by:
- **False sharing** at K/T < 64 (adjacent threads' data on same cache line)
- **Memory bandwidth** at large N*K (all threads share L3/DRAM bus)

Group-parallel scaling is limited by:
- **Barrier cost** between stages (~100-500ns per barrier, 3-4 barriers per plan)
- **Group imbalance** when num_groups is not divisible by T

## Architectural Insight

The key advantage is a design decision made at the start: **split-complex layout with K as the innermost dimension**. This layout naturally supports K-slicing because each thread's data is contiguous in memory. The permutation-free roundtrip architecture means there's no global data shuffle between stages — each stage's groups are independent within the K slice.

MKL and FFTW use interleaved complex or different memory layouts that make batch-parallel slicing harder. Their threading was designed for single large transforms (N >> 1, K=1), not for batched workloads.

For batched DSP applications (the primary VectorFFT use case), this architectural choice gives a fundamental threading advantage that cannot be matched by tuning MKL's parameters.
