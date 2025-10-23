# Threading Architecture

## Overview

This code will use a **hierarchical plan-based architecture** where threading is abstracted into a separate module that can work with different threading backends.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Application                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   FFTW Planning System                       │
│  • Analyzes problem size                                     │
│  • Tests multiple strategies (radixes, threading)            │
│  • Benchmarks and selects optimal plan                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Plan Hierarchy                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Main Plan (P)                                       │   │
│  │  • nthr: number of threads                           │   │
│  │  • r: radix                                          │   │
│  │  • cld: child plan (sequential)                      │   │
│  │  • cldws[]: child worker plans (parallel)            │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                Threading Abstraction Layer                   │
│              X(spawn_loop)(nthr, nthr, fn, data)             │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   pthreads   │  │   OpenMP     │  │   Windows    │
│   Backend    │  │   Backend    │  │   Threads    │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Execution Flow

### 1. Planning Phase (mkplan)

```
mkplan(problem, planner)
    │
    ├─> Calculate work distribution
    │   • n = problem size
    │   • r = choose_radix(n)
    │   • m = n / r (sub-problems)
    │   • block_size = m / nthr
    │   • nthr = actual threads needed
    │
    ├─> Create worker plans (parallel)
    │   for each thread i:
    │       cldws[i] = create_worker_plan(
    │           start: i * block_size,
    │           count: block_size
    │       )
    │
    └─> Create main plan (sequential)
        cld = create_main_plan(...)
```

### 2. Execution Phase (apply)

#### DIT (Decimation in Time) Strategy:
```
apply_dit(input, output)
    │
    ├─> Step 1: Sequential main transform
    │   cld->apply(input, output)
    │   • Rearranges data
    │   • Prepares for parallel phase
    │
    └─> Step 2: Parallel worker transforms
        spawn_loop(nthr, spawn_apply, data)
            ├─> Thread 0: cldws[0]->apply(block_0)
            ├─> Thread 1: cldws[1]->apply(block_1)
            ├─> Thread 2: cldws[2]->apply(block_2)
            └─> Thread n: cldws[n]->apply(block_n)
        [Wait for all threads to complete]
```

#### DIF (Decimation in Frequency) Strategy:
```
apply_dif(input, output)
    │
    ├─> Step 1: Parallel worker transforms
    │   spawn_loop(nthr, spawn_apply, data)
    │       ├─> Thread 0: cldws[0]->apply(block_0)
    │       ├─> Thread 1: cldws[1]->apply(block_1)
    │       └─> Thread n: cldws[n]->apply(block_n)
    │   [Wait for all threads to complete]
    │
    └─> Step 2: Sequential main transform
        cld->apply(input, output)
        • Combines results
        • Produces final output
```

## Key Data Structures

### Plan Structure (P)
```c
typedef struct {
    plan_rdft super;      // Base plan interface
    plan *cld;            // Child plan (sequential part)
    plan **cldws;         // Array of worker plans (parallel)
    int nthr;             // Number of threads
    INT r;                // Radix used
} P;
```

### Spawn Data
```c
typedef struct {
    plan **cldws;         // Worker plans
    R *IO;                // Data buffer
} PD;

// Each thread receives:
typedef struct {
    int thr_num;          // Thread ID (0 to nthr-1)
    void *data;           // Points to PD structure
} spawn_data;
```

## Work Distribution Strategy

### Block-Based Parallelism
```
Problem size: n = 1024
Radix: r = 4
Sub-problems: m = n/r = 256
Threads: nthr = 4

Block size = ⌈m / nthr⌉ = ⌈256 / 4⌉ = 64

Thread 0: processes indices [0,   63]   (64 items)
Thread 1: processes indices [64,  127]  (64 items)
Thread 2: processes indices [128, 191]  (64 items)
Thread 3: processes indices [192, 255]  (64 items)

Each thread works on independent data - NO SYNCHRONIZATION needed!
```

### Real DFT Special Case
```c
// For R2HC/HC2R transforms, only half the spectrum is computed
mcount = (m + 2) / 2;
block_size = (mcount + plnr->nthr - 1) / plnr->nthr;
nthr = (mcount + block_size - 1) / block_size;

// Last thread may have fewer items:
if (i == nthr - 1)
    count = mcount - i*block_size;  // Remainder
else
    count = block_size;
```

## Threading Backends

### Spawn Loop Interface
```c
void X(spawn_loop)(
    int loopmax,           // Number of iterations
    int nthr,              // Number of threads to use
    void *(*spawn_apply)(spawn_data *),  // Worker function
    void *data             // Data passed to workers
);
```

### Backend Implementations

**pthreads:**
```c
// Create thread pool
pthread_t threads[nthr];
for (i = 0; i < nthr; i++) {
    spawn_data d = { .thr_num = i, .data = data };
    pthread_create(&threads[i], NULL, spawn_apply, &d);
}
// Wait for completion
for (i = 0; i < nthr; i++) {
    pthread_join(threads[i], NULL);
}
```

**OpenMP:**
```c
#pragma omp parallel for num_threads(nthr)
for (i = 0; i < nthr; i++) {
    spawn_data d = { .thr_num = i, .data = data };
    spawn_apply(&d);
}
```

## Optimization Features

### 1. Adaptive Thread Count
```c
// Don't create more threads than work items
nthr = min(requested_threads, number_of_blocks);

// If work is too small, use fewer threads
if (block_size < THRESHOLD)
    nthr = 1;  // Fall back to sequential
```

### 2. Nested Planning
```c
// Save outer thread count
plnr_nthr_save = plnr->nthr;

// Allocate threads to nested plans
plnr->nthr = (plnr->nthr + nthr - 1) / nthr;

// Create nested plans with reduced thread count
cld = mkplan(plnr, subproblem);

// Restore thread count
plnr->nthr = plnr_nthr_save;
```

### 3. Plan Reuse
```c
// Detect identical worker plans to save memory
for (i = 0; i < nthr; ++i) {
    if (i == 0 || 
        (ego->cldws[i] != ego->cldws[i-1] &&
         (i <= 1 || ego->cldws[i] != ego->cldws[i-2])))
    {
        // Only print/store unique plans
    }
}
```

## Memory Access Pattern

### Cache-Friendly Design
```
Original array: [a0, a1, a2, ..., a255]

After main transform (DIT):
┌─────────────────────────────────────┐
│  Rearranged for parallel access     │
│  [block0][block1][block2][block3]   │
└─────────────────────────────────────┘

Each thread accesses contiguous memory:
Thread 0 → [a0  ... a63 ]  ← Cache-friendly
Thread 1 → [a64 ... a127]  ← No false sharing
Thread 2 → [a128... a191]
Thread 3 → [a192... a255]
```

## Complete Example: 4-Thread DIT Execution

```
Input: 1024 complex numbers
Radix: 4
Threads: 4

Step 1: SEQUENTIAL - Main Transform
    ┌──────────────────────────────────┐
    │  Rearrange 1024 → 4 × 256 blocks │
    │  Apply radix-4 butterflies       │
    └──────────────────────────────────┘
    Time: T₁

Step 2: PARALLEL - Worker Transforms
    Thread 0: FFT on block 0 (64 of 256 items)  ┐
    Thread 1: FFT on block 1 (64 of 256 items)  ├─ Time: T₂
    Thread 2: FFT on block 2 (64 of 256 items)  │
    Thread 3: FFT on block 3 (64 of 256 items)  ┘

Total time ≈ T₁ + T₂ (speedup depends on T₁/T₂ ratio)
```

## Solver Registration

```c
// Create solver for specific radix and decomposition
solver = X(mksolver_ct_threads)(
    sizeof(ct_solver),     // Solver size
    r,                     // Radix (2, 3, 4, 5, ...)
    DECDIT,                // Decomposition type
    mkcldw,                // Worker plan factory
    force_vrecursion       // Vector recursion strategy
);

// Register with planner
REGISTER_SOLVER(planner, solver);
```

The planner will try different solvers and pick the fastest!

## Thread Safety

- **Planning**: NOT thread-safe (use locks if planning in parallel)
- **Execution**: Thread-safe (plans are read-only during execution)
- **Worker plans**: Independent (no shared state between threads)

## Performance Characteristics

**Pros:**
- Zero synchronization during parallel phase
- Cache-friendly memory access
- Scales well up to ~8-16 threads
- Works with any radix

**Cons:**
- Sequential bottleneck (main transform)
- Amdahl's Law limits: speedup ≤ 1/(sequential_fraction)
- Small problems don't benefit from threading
- Thread creation overhead for very small FFTs


The threading is completely transparent to the user!