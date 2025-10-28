# Bluestein Memory Management Architecture Report

## Executive Summary

Bluestein's algorithm transforms arbitrary-size FFTs into circular convolutions computed via power-of-2 FFTs. This implementation employs a **three-tier memory hierarchy**: precomputed plan data (owned), recursive FFT sub-plans (owned), and user-provided workspace (borrowed). The architecture solves the memory challenge through explicit separation of planning-time allocation and execution-time workspace, enabling thread-safe plan reuse while maintaining optimal cache performance.

---

## Problem Space: Memory Requirements for Arbitrary-Size FFTs

### Fundamental Challenge

For an arbitrary size N, Bluestein's algorithm requires:

1. **Chirp twiddle factors**: N complex numbers storing exp(±πin²/N)
2. **Kernel FFT precomputation**: M complex numbers (where M = next_pow2(2N-1))
3. **Execution workspace**: 3M complex numbers for intermediate computations
4. **Recursive FFT infrastructure**: Two complete FFT plans for size M (forward + inverse)

The core challenge: **M can be significantly larger than N**. For example:
- N = 509 (prime) → M = 1024 (2.01× overhead)
- N = 1021 (prime) → M = 2048 (2.00× overhead)
- N = 1000 → M = 2048 (2.05× overhead)

This padding is mathematically necessary to prevent circular convolution wraparound, but creates substantial memory pressure.

---

## Architecture Overview: Separation of Concerns

### Three-Phase Memory Model

The implementation separates memory responsibilities into three distinct phases:

```
PLANNING TIME (fft_init)
    ↓
[Allocate Plan Structure] ← One-time cost, amortized
    ↓
[Precompute Chirp & Kernel] ← Owned by plan, freed with plan
    ↓
[Create Recursive FFT Plans] ← Owned by plan, recursive fft_init()
    ↓
EXECUTION TIME (fft_exec_dft)
    ↓
[User Provides Workspace] ← Borrowed, not owned
    ↓
[Execute with Precomputed Data] ← Zero allocation, pure computation
```

This separation enables **plan reuse**: expensive precomputation happens once, fast execution happens many times with different workspace buffers.

---

## Requirement 1: Precomputed Plan Data Storage

### The Need

Bluestein requires two precomputed arrays that depend only on N and direction:

1. **Chirp sequence**: `exp(sign × πin²/N)` for n = 0..N-1
   - Sign: +1 for forward, -1 for inverse
   - Size: N complex numbers (16N bytes)

2. **Kernel FFT**: FFT of chirp conjugate, zero-padded and mirrored
   - Construction: `[1, chirp[1]*, ..., chirp[N-1]*, 0, ..., 0, chirp[N-1]*, ..., chirp[1]*]`
   - Size: M complex numbers (16M bytes)
   - Purpose: Pointwise multiplication in frequency domain

### Solution: Direction-Specific Opaque Plans

The implementation uses **separate structure types**:

```c
struct bluestein_plan_forward_s {
    int N;
    int M;
    
    fft_data *chirp_forward;       // Owned: +π sign baked in
    fft_data *kernel_fft_forward;  // Owned: precomputed FFT
    
    fft_object fft_plan_m;         // Owned: recursive plan
    fft_object ifft_plan_m;        // Owned: recursive plan
};

struct bluestein_plan_inverse_s {
    int N;
    int M;
    
    fft_data *chirp_inverse;       // Owned: -π sign baked in
    fft_data *kernel_fft_inverse;  // Owned: precomputed FFT
    
    fft_object fft_plan_m;         // Owned: recursive plan
    fft_object ifft_plan_m;        // Owned: recursive plan
};
```

**Method: Aligned Allocation at Plan Creation**

```c
// In bluestein_plan_create_forward():
plan->chirp_forward = compute_forward_chirp(N);
// → Allocates aligned_alloc(32, N * sizeof(fft_data))
// → AVX2-friendly alignment for vectorized access

plan->kernel_fft_forward = compute_forward_kernel_fft(chirp, N, M);
// → Allocates aligned_alloc(32, M * sizeof(fft_data))
// → Computes FFT immediately, stores result
```

**Memory Footprint Per Plan:**
- Chirp: 16N bytes (aligned to 32)
- Kernel FFT: 16M bytes (aligned to 32)
- Structure overhead: ~64 bytes
- **Total: 16(N + M) + 64 bytes**

For N = 509 (prime):
- M = 1024
- Chirp: 8,144 bytes (~8 KB)
- Kernel: 16,384 bytes (16 KB)
- **Total: ~24 KB per direction**

---

## Requirement 2: Recursive FFT Plan Storage

### The Need

Bluestein's execution requires **three FFTs of size M**:

1. FFT of chirp-modulated input → frequency domain
2. IFFT of pointwise product → time domain (convolution result)
3. (Implicit in kernel precomputation)

Each FFT operation needs its own plan structure with twiddle factors.

### Solution: Cached Internal Plans

The implementation maintains a **global cache** of power-of-2 plans:

```c
static fft_object internal_fft_cache[2][32] = {NULL};
//                                    ↑   ↑
//                              direction log2(M)

static fft_object get_internal_fft_plan(int M, fft_direction_t direction)
{
    int log2_M = __builtin_ctz(M);  // M guaranteed power-of-2
    int dir_idx = (direction == FFT_FORWARD) ? 0 : 1;
    
    if (!internal_fft_cache[dir_idx][log2_M]) {
        internal_fft_cache[dir_idx][log2_M] = fft_init(M, direction);
        // ↑ Recursive call, guaranteed to select INPLACE_BITREV
    }
    
    return internal_fft_cache[dir_idx][log2_M];
}
```

**Method: Lazy Initialization with Sharing**

When creating a Bluestein plan:

```c
plan->fft_plan_m = get_internal_fft_plan(M, FFT_FORWARD);
plan->ifft_plan_m = get_internal_fft_plan(M, FFT_INVERSE);
plan->plans_are_cached = 1;  // Mark as borrowed, not owned
```

**Key Design Decision: Plans are BORROWED, not owned**

```c
void bluestein_plan_free_forward(bluestein_plan_forward *plan) {
    if (!plan->chirp_is_cached)
        free(plan->chirp_forward);     // Free owned chirp
    free(plan->kernel_fft_forward);     // Free owned kernel
    // Note: fft_plan_m NOT freed (borrowed from cache)
    free(plan);
}
```

**Memory Footprint: Cached Plans**

For M = 1024 (power-of-2), a Cooley-Tukey plan needs:
- Twiddle factors: (M-1) complex numbers = 16,368 bytes
- Stage descriptors: 10 stages × ~48 bytes = 480 bytes
- **Total per direction: ~16 KB**

Cache stores up to 32 sizes × 2 directions = 64 plans maximum.

**Sharing Benefits:**
- Multiple N values map to same M (e.g., N=509,510,511 → M=1024)
- Plans reused across different Bluestein plans
- One-time allocation cost per M, not per N

---

## Requirement 3: Execution Workspace

### The Need

Bluestein's algorithm has five computational steps:

```
Step 1: Multiply input by chirp + zero-pad
        → Requires: M-element buffer (buffer_a)

Step 2: FFT of modulated input
        → Requires: M-element buffer (buffer_b)

Step 3: Pointwise multiply with kernel FFT
        → Requires: M-element buffer (buffer_c)

Step 4: IFFT of product
        → Reuses: buffer_b (output of IFFT)

Step 5: Multiply by chirp again
        → Writes directly to user's output array
```

**Minimum workspace: 3M complex numbers** (buffers A, B, C)

For N = 509 → M = 1024:
- **Workspace: 3 × 1024 × 16 bytes = 49,152 bytes (~48 KB)**

### Solution: User-Provided Workspace

The implementation **never allocates execution buffers internally**:

```c
int bluestein_exec_forward(
    bluestein_plan_forward *plan,
    const fft_data *input,
    fft_data *output,
    fft_data *scratch,        // User must provide
    size_t scratch_size)      // Safety check
{
    if (scratch_size < 3 * M) return -1;  // Guard against undersized buffer
    
    // Partition user's buffer into three zones
    fft_data *buffer_a = scratch;
    fft_data *buffer_b = scratch + M;
    fft_data *buffer_c = scratch + 2 * M;
    
    // Proceed with algorithm using partitioned workspace...
}
```

**Method: Query-Allocate-Execute Pattern**

Users follow this protocol:

```c
// Step 1: Query required workspace size
size_t workspace_size = fft_get_workspace_size(plan);
// → Delegates to bluestein_get_scratch_size(N)
// → Returns 3 × next_pow2(2N-1)

// Step 2: Allocate workspace (user's responsibility)
fft_data *workspace = aligned_alloc(32, workspace_size * sizeof(fft_data));

// Step 3: Execute with workspace
fft_exec_dft(plan, input, output, workspace);
// → Calls bluestein_exec_forward/inverse internally
// → workspace partitioned into 3 buffers

// Step 4: Free workspace (user's responsibility)
aligned_free(workspace);
```

**Thread Safety Implications:**

Because workspace is provided per-call:
- **One plan, many threads**: Each thread provides its own workspace
- **No mutex needed during execution**: Plan is read-only after creation
- **Stack allocation possible**: For small N, workspace can be VLA

Example with stack allocation:
```c
if (workspace_size < 1024) {
    fft_data workspace[workspace_size];  // VLA on stack
    fft_exec_dft(plan, input, output, workspace);
}
```

---

## Requirement 4: Integration with Parent FFT Plan

### The Need

From the parent library's perspective, Bluestein is one of three execution strategies:

```c
typedef enum {
    FFT_EXEC_INPLACE_BITREV,    // 0 workspace
    FFT_EXEC_STOCKHAM,          // N workspace
    FFT_EXEC_BLUESTEIN,         // 3M workspace ← Different size!
} fft_exec_strategy_t;
```

The parent must:
1. Store the Bluestein sub-plan
2. Query correct workspace size
3. Dispatch to Bluestein execution path

### Solution: Union for Type-Safe Storage

The parent `fft_plan` structure uses a union:

```c
typedef struct fft_plan_struct {
    int n_input;                    // User's requested N
    int n_fft;                      // Actual FFT size (M for Bluestein)
    fft_direction_t direction;
    fft_exec_strategy_t strategy;
    
    // Cooley-Tukey data (unused for Bluestein)
    int num_stages;
    int factors[MAX_FFT_STAGES];
    stage_descriptor stages[MAX_FFT_STAGES];
    
    // Bluestein data (only one active)
    union {
        bluestein_plan_forward *bluestein_fwd;   // if direction == FFT_FORWARD
        bluestein_plan_inverse *bluestein_inv;   // if direction == FFT_INVERSE
        void *bluestein_generic;                 // for NULL checks
    };
} fft_plan;
```

**Method: Direction-Based Dispatch in Planner**

```c
// In fft_planner.c → plan_bluestein():
if (direction == FFT_FORWARD) {
    plan->bluestein_fwd = bluestein_plan_create_forward(N);
    // union member: bluestein_fwd active
} else {
    plan->bluestein_inv = bluestein_plan_create_inverse(N);
    // union member: bluestein_inv active
}

plan->n_input = N;      // Original size
plan->n_fft = M;        // Padded size (for workspace calculation)
plan->strategy = FFT_EXEC_BLUESTEIN;
```

**Method: Workspace Size Query**

```c
// In fft_planner.c → fft_get_workspace_size():
size_t fft_get_workspace_size(fft_object plan) {
    switch (plan->strategy) {
        case FFT_EXEC_INPLACE_BITREV:
            return 0;
        
        case FFT_EXEC_STOCKHAM:
            return (size_t)plan->n_fft;  // N elements
        
        case FFT_EXEC_BLUESTEIN:
            return bluestein_get_scratch_size(plan->n_input);
            // → Returns 3 × next_pow2(2N-1)
    }
}
```

**Method: Execution Dispatch**

```c
// In fft_execute.c → fft_exec_dft():
switch (plan->strategy) {
    case FFT_EXEC_BLUESTEIN:
        if (plan->direction == FFT_FORWARD) {
            return bluestein_exec_forward(
                plan->bluestein_fwd,  // Correct union member
                input, output, workspace, scratch_size
            );
        } else {
            return bluestein_exec_inverse(
                plan->bluestein_inv,  // Correct union member
                input, output, workspace, scratch_size
            );
        }
}
```

---

## Memory Ownership Summary

### Planning-Time Allocations (Owned by Plan)

| Component | Size | Lifetime | Freed By |
|-----------|------|----------|----------|
| `chirp_forward/inverse` | 16N bytes | Plan lifetime | `bluestein_plan_free_*()` |
| `kernel_fft_forward/inverse` | 16M bytes | Plan lifetime | `bluestein_plan_free_*()` |
| Plan structure | ~64 bytes | Plan lifetime | `bluestein_plan_free_*()` |

**Total per Bluestein plan: 16(N + M) + 64 bytes**

### Cached Allocations (Shared, Not Owned)

| Component | Size | Lifetime | Freed By |
|-----------|------|----------|----------|
| `internal_fft_cache` plans | ~16 KB per (M, direction) | Program lifetime | Never (global cache leak) |

**Note:** The code has a cache cleanup gap—internal FFT plans are never freed. This is a minor leak for long-running programs, but negligible since cache size is bounded (64 plans × ~16 KB = ~1 MB maximum).

### Execution-Time Allocations (Borrowed from User)

| Component | Size | Lifetime | Freed By |
|-----------|------|----------|----------|
| Workspace buffer | 3M × 16 bytes | Per-call | User |

**User's responsibility:** Allocate before `fft_exec_dft()`, free after.

---

## Performance Characteristics

### Memory Access Pattern During Execution

**Step 1: Input Chirp Multiplication (Streaming)**
```c
for (n = 0; n < N; n++) {
    buffer_a[n] = complex_mul(input[n], chirp[n]);
}
memset(buffer_a + N, 0, (M - N) * sizeof(fft_data));
```
- **Access pattern**: Sequential reads (input, chirp), sequential writes (buffer_a)
- **Cache behavior**: Excellent (all streaming, no random access)
- **Vectorization**: AVX2 processes 2 complex numbers per iteration

**Step 2-4: FFT Operations (Cached Sub-Plans)**
```c
fft_exec(plan->fft_plan_m, buffer_a, buffer_b);     // M-point FFT
// Pointwise multiply: buffer_b × kernel_fft → buffer_c
fft_exec(plan->ifft_plan_m, buffer_c, buffer_b);    // M-point IFFT
```
- **Access pattern**: Determined by power-of-2 FFT (bit-reversal + butterfly)
- **Cache behavior**: Good (M is power-of-2, stage-friendly)
- **Twiddle reuse**: Sub-plans precomputed, zero overhead

**Step 5: Output Chirp Multiplication (Streaming)**
```c
for (k = 0; k < N; k++) {
    output[k] = complex_mul(buffer_b[k], chirp[k]);
}
```
- **Access pattern**: Sequential reads (buffer_b, chirp), sequential writes (output)
- **Cache behavior**: Excellent (streaming)

### Workspace Reuse Analysis

The three buffers serve distinct purposes:

- **buffer_a**: Used in Step 1, dead after Step 2
- **buffer_b**: Written in Step 2, read/written in Steps 3-4, read in Step 5
- **buffer_c**: Written in Step 3, read in Step 4, dead after

**Could optimize to 2M workspace?** No, because:
- Step 3 reads buffer_b (output of Step 2)
- Step 3 writes buffer_c (input to Step 4)
- Cannot alias without destroying buffer_b before Step 4

The 3M requirement is **minimal and unavoidable** for the algorithm structure.

---

## Comparison with Alternative Architectures

### Alternative 1: Allocate Workspace Internally

**Rejected approach:**
```c
int bluestein_exec_forward(...) {
    fft_data *workspace = malloc(3 * M * sizeof(fft_data));
    // Execute algorithm
    free(workspace);
}
```

**Why rejected:**
- **Allocation overhead**: malloc/free on every call (~1-10 µs)
- **Thread contention**: Memory allocator mutex under load
- **Cache pollution**: Repeatedly allocating/freeing same size
- **Unpredictable latency**: Fragmentation effects

**Chosen approach benefits:**
- User can pre-allocate workspace once, reuse across calls
- Thread-local workspace eliminates allocator contention
- Stack allocation possible for small transforms

### Alternative 2: Store Workspace in Plan

**Rejected approach:**
```c
struct bluestein_plan_forward_s {
    // ... existing fields ...
    fft_data *workspace;  // Allocated during plan creation
};
```

**Why rejected:**
- **Thread safety destroyed**: Cannot share plan across threads
- **Memory waste**: Workspace sits idle between calls
- **Inflexible**: User may want different workspace per thread

**Chosen approach benefits:**
- Plans are immutable after creation → thread-safe reads
- Memory only allocated when actually executing
- User controls allocation strategy (heap, stack, mmap, etc.)

---

## Failure Modes and Error Handling

### Insufficient Workspace

```c
if (scratch_size < 3 * M) {
    return -1;  // Guard against buffer overflow
}
```

**User error:** Queried size incorrectly or used wrong plan
**Impact:** Detected before memory corruption
**Recovery:** User must allocate correct size and retry

### Allocation Failures During Planning

```c
plan->chirp_forward = compute_forward_chirp(N);
if (!plan->chirp_forward) {
    free(plan);
    return NULL;
}
```

**System error:** Out of memory during `aligned_alloc()`
**Impact:** Planning fails, returns NULL
**Recovery:** User must handle NULL plan (reduce N or free memory)

### Recursive Plan Creation Failure

```c
plan->fft_plan_m = get_internal_fft_plan(M, FFT_FORWARD);
if (!plan->fft_plan_m) {
    free(plan->chirp_forward);
    free(plan);
    return NULL;
}
```

**Should never occur:** M is power-of-2, guaranteed to succeed
**Defensive check:** Handles unexpected fft_init() failure
**Recovery:** Clean up partial plan, propagate error to caller

---

## Conclusion

This Bluestein implementation solves the arbitrary-size FFT memory challenge through **explicit separation of planning and execution responsibilities**:

1. **Plan owns precomputed data**: Chirps and kernel FFTs allocated once, reused forever
2. **Plan borrows recursive FFT infrastructure**: Global cache prevents duplicate allocations
3. **User provides execution workspace**: Enables thread safety and flexible memory management

The architecture achieves:
- **Predictable memory footprint**: 16(N+M) bytes per plan + 48M bytes per thread
- **Thread-safe plan reuse**: Zero synchronization overhead during execution
- **Cache-friendly execution**: Sequential access patterns, precomputed twiddles
- **Flexible allocation**: Users choose heap, stack, or custom allocators

Total memory for N=509:
- **Planning:** ~24 KB (chirp + kernel) + ~32 KB (cached M=1024 plans) = **56 KB one-time**
- **Execution:** ~48 KB workspace per thread (user-managed)

The design trades planning-time cost (precomputation) for execution-time performance (zero allocation), consistent with the FFTW philosophy of "compile once, execute many."