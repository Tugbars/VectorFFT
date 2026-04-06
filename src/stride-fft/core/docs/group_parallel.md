# Group-Parallel Threading in VectorFFT

## The Problem: False Sharing in K-Split

VectorFFT stores batched FFT data in split-complex layout:

```
re[n * K + k]   for n = 0..N-1, k = 0..K-1
```

The naive threading approach (K-split) divides the K batch dimension across T threads. Thread t processes lanes `[t*S, (t+1)*S)` where S = K/T:

```
Thread 0: re[n*K + 0   .. n*K + S-1]
Thread 1: re[n*K + S   .. n*K + 2S-1]
Thread 2: re[n*K + 2S  .. n*K + 3S-1]
...
```

At K=256, T=8: each thread gets S=32 doubles = 256 bytes = 4 cache lines. The butterfly codelet reads and writes R legs at stride `S_stage * K`, where each leg touches these same 4 cache lines. Adjacent threads' slices sit on adjacent (or the same) cache lines.

When two cores write to addresses on the same 64-byte cache line, the hardware cache coherency protocol (MESI) forces the line to bounce between cores. This is **false sharing** — the cores aren't accessing the same data, but the cache hardware doesn't know that.

Result: K=256 T=8 gives only **2.2x** speedup instead of the expected ~6-8x.

## The Solution: Group-Parallel Execution

Instead of splitting the K dimension, split the **groups** within each stage.

A stride-based FFT decomposes N into stages. Stage s with radix R has `N/R` groups, where each group is an independent R-point butterfly operating on K elements. The key insight: **groups within the same stage are independent** — they read and write disjoint memory regions.

```
Stage s (R=5, N=1000):
  Group 0:  butterfly on legs [base+0, base+stride, ..., base+4*stride]
  Group 1:  butterfly on legs [base+K, base+K+stride, ..., base+K+4*stride]
  ...
  Group 199: independent of all other groups
```

Group-parallel assigns groups to threads:

```
Thread 0: groups [0, 25)      — processes all K=256 lanes for each group
Thread 1: groups [25, 50)     — same K, different groups
...
Thread 7: groups [175, 200)
```

Each thread runs the full codelet with K=256 lanes — full SIMD utilization, no false sharing. Different groups touch different memory regions (separated by at least K doubles = 2KB), so no cache line contention.

## The Cost: Inter-Stage Barriers

Groups within a stage are independent, but **stages depend on each other**. Stage s+1's groups read data that stage s wrote. All threads must finish stage s before any thread starts stage s+1.

This requires a barrier between stages:

```
for each stage s:
    parallel: each thread processes its groups with full K
    barrier: wait for all threads
```

VectorFFT uses a **sense-reversing spin barrier** — threads spin on a shared flag that flips each generation. Cost: ~100-500ns per barrier. For a 4-stage plan at ~500us total, the barrier overhead is < 0.5%.

## Strategy Selection

VectorFFT automatically selects the best strategy per call:

| Condition | Strategy | Reason |
|---|---|---|
| K/T >= 256 | K-split | Each thread gets ≥2KB per group — no false sharing, zero barriers |
| K/T < 256 | Group-parallel | Full K per codelet call, barriers between stages |
| T = 1 | Single-threaded | No overhead |

The threshold (256 doubles = 2KB = 32 cache lines) was determined empirically on Intel Raptor Lake (i9-14900KF).

## Results

N=1000, K=256, T=8 (8 P-cores, i9-14900KF):

| Strategy | Time (ns) | Speedup |
|---|---|---|
| Single-threaded | 478,340 | 1.00x |
| K-split | 222,168 | 2.15x |
| Group-parallel | 112,601 | **4.25x** |

Group-parallel nearly doubles the effective speedup compared to K-split for small K.

## Implementation Details

**Barrier**: Sense-reversing with atomic increment (`InterlockedIncrement` on Win32, `__sync_add_and_fetch` on Linux). Last thread to arrive flips the sense flag, releasing all waiters. No OS primitives — pure spin-wait.

**Group assignment**: Static partitioning: thread t processes groups `[ng*t/T, ng*(t+1)/T)`. Balanced within ±1 group.

**Full codelet paths**: The group-parallel executor supports all twiddle paths (t1s scalar broadcast, temp buffer, legacy t1, n1 fallback) — each thread executes the same logic as single-threaded, just on fewer groups.

**Thread pool**: Persistent spin-waiting workers (created once by `stride_set_num_threads`). Dispatch is a single volatile write (clear the done flag) — ~10ns wake latency.
